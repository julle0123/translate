"""번역 노드 - LangGraph 노드에서 사용하는 번역 함수 """
import os
import asyncio
import re
import logging
import time
from asyncio import Queue
from typing import Dict, Any, List, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.callbacks import AsyncCallbackHandler
from transformers import AutoTokenizer

from state.translation_state import TranslationState
from prompt.prompts import create_translation_prompt
from utils.language_utils import get_language_name
from schema.schemas import ServiceInfo, UserInfo

# 로거 설정
LOGGER = logging.getLogger(__name__)
# 로거 레벨 및 핸들러 설정 (터미널 출력용)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class TranslateAgent:
    """번역 에이전트 클래스 """
    
    def __init__(
        self,
        service_info: Optional[Dict[str, Any]] = None,
        chat_req: Optional[Dict[str, Any]] = None,
        llm: Optional[ChatOpenAI] = None,
        callbacks: Optional[List] = None
    ):
        """
        초기화 
        
        Args:
            service_info: 서비스 정보
            chat_req: 채팅 요청 정보
            llm: LLM 인스턴스
            callbacks: 콜백 리스트
        """
        self.agent_name = "TranslateAgent"
        self.service_info = service_info
        self.chat_req = chat_req
        self.llm = llm or ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            streaming=True,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.callbacks = callbacks or []
        
        # 설정값 (환경변수 우선, 없으면 기본값)
        # chunk_size와 chunk_overlap은 토큰 수로 해석됨
        self.chunk_size = int(os.getenv("TRANSLATE_CHUNK_SIZE", "1000"))  # 기본값을 토큰 수로 변경 (1000 토큰)
        self.chunk_overlap = int(os.getenv("TRANSLATE_CHUNK_OVERLAP", "0"))  # 오버랩 없음
        self.max_concurrent = int(os.getenv("TRANSLATE_MAX_CONCURRENT", "3"))
        
        # 토큰 사용량 추적 (이미지 코드와 동일한 변수명)
        self.crt_step_inpt_tokn_cnt = 0
        self.crt_step_otpt_tokn_cnt = 0
        self.crt_step_totl_tokn_cnt = 0
        
        # transformers AutoTokenizer 초기화 (모델에 맞는 토크나이저 사용)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            # OpenAI 모델에 대응하는 Hugging Face 토크나이저 매핑
            # gpt-4o, gpt-4o-mini는 o200k_base와 유사한 토크나이저 사용
            # OpenAI 모델은 Hugging Face에 직접 매핑되지 않으므로, 유사한 토크나이저 사용
            if "gpt-4o" in model_name.lower():
                # o200k_base와 유사한 토크나이저 (GPT-2 기반 또는 cl100k_base 유사)
                tokenizer_name = "gpt2"  # 또는 "Xenova/gpt-4o-mini" 등이 있다면 사용
            elif "gpt-4" in model_name.lower() or "gpt-3.5" in model_name.lower():
                # cl100k_base와 유사한 토크나이저
                tokenizer_name = "gpt2"
            else:
                tokenizer_name = "gpt2"  # 기본값
            
            # 환경 변수로 토크나이저 이름을 지정할 수 있음
            tokenizer_name = os.getenv("TOKENIZER_NAME", tokenizer_name)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            LOGGER.info(f"토크나이저 초기화 완료: {tokenizer_name}")
        except Exception as e:
            LOGGER.warning(f"토크나이저 초기화 실패, gpt2 사용: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception as e2:
                LOGGER.error(f"기본 토크나이저 초기화도 실패: {e2}")
                # 폴백: 간단한 문자 수 기반 계산 (토큰 수 근사치)
                self.tokenizer = None
        
        LOGGER.info(f"TranslateAgent 초기화: chunk_size={self.chunk_size}토큰, chunk_overlap={self.chunk_overlap}토큰, max_concurrent={self.max_concurrent}")
    
    def _split_text_by_tokens(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int = 0
    ) -> List[str]:
        """
        토큰 수 기반으로 텍스트를 정확하게 청크로 분할
        
        Args:
            text: 분할할 텍스트
            chunk_size: 각 청크의 최대 토큰 수
            chunk_overlap: 청크 간 오버랩 토큰 수
        
        Returns:
            List[str]: 분할된 청크 리스트
        """
        if self.tokenizer is None:
            # 토크나이저가 없으면 문자 수 기반으로 폴백
            LOGGER.warning("토크나이저가 없어 문자 수 기반으로 분할합니다.")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 4,  # 대략적인 변환 (토큰 1개 ≈ 4자)
                chunk_overlap=chunk_overlap * 4,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return splitter.split_text(text)
        
        # 빈 텍스트 처리
        if not text or not text.strip():
            return [text] if text else [""]
        
        # 텍스트를 토큰으로 인코딩
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)
        
        if total_tokens <= chunk_size:
            return [text]
        
        chunks = []
        start_idx = 0
        prev_start_idx = -1  # 무한 루프 방지를 위한 이전 start_idx 추적
        
        # 구분자 우선순위 (다국어 지원)
        # 영어, 한국어, 일본어, 중국어 등 다양한 언어의 구분자 포함
        separators = [
            "\n\n",  # 단락 구분
            "\n",    # 줄바꿈
            ". ",    # 영어 문장 종료
            "。",    # 일본어/중국어 문장 종료
            "！",    # 일본어/중국어 감탄
            "？",    # 일본어/중국어 의문
            "! ",    # 영어 감탄
            "? ",    # 영어 의문
            ".",     # 영어 문장 종료 (공백 없음)
            "!",     # 영어 감탄 (공백 없음)
            "?",     # 영어 의문 (공백 없음)
            " ",     # 공백 (공백이 있는 언어용)
            ""       # 최후의 수단
        ]
        separator_token_ids = []
        for sep in separators:
            if sep:
                sep_tokens = self.tokenizer.encode(sep, add_special_tokens=False)
                if sep_tokens:
                    separator_token_ids.append((sep, sep_tokens))
        
        # 최소 청크 크기 (너무 작은 청크 방지)
        # chunk_size가 1이면 min_chunk_size도 1, 그 외에는 50% 이상
        min_chunk_size = max(1, chunk_size // 2) if chunk_size > 1 else 1
        
        while start_idx < total_tokens:
            # 현재 청크의 끝 인덱스 계산
            end_idx = min(start_idx + chunk_size, total_tokens)
            
            # 무한 루프 방지: 최소 1토큰은 전진해야 함
            if end_idx <= start_idx:
                end_idx = start_idx + 1
            
            # 청크 크기를 초과하지 않는 범위에서 구분자 찾기
            if end_idx < total_tokens:
                # 끝에서부터 구분자를 찾아서 자연스러운 경계로 조정
                best_split_idx = end_idx
                # 검색 범위: 최소 청크 크기를 보장하면서 가능한 한 뒤로 검색
                # search_start는 start_idx보다 크고 end_idx보다 작아야 함
                # 검색 범위: 최소 min_chunk_size 이상, 최대 end_idx - 1까지
                min_search_start = max(start_idx + 1, start_idx + min_chunk_size)
                max_search_start = end_idx - 1
                
                # 선호하는 검색 시작점: 끝에서 50% 지점
                preferred_start = max(min_search_start, end_idx - chunk_size // 2)
                search_start = min(preferred_start, max_search_start)
                
                # 최종 검증: search_start가 유효한 범위에 있는지 확인
                if search_start <= start_idx:
                    search_start = min(start_idx + 1, end_idx - 1)
                if search_start >= end_idx:
                    search_start = max(start_idx + 1, end_idx - 1)
                
                # search_start가 유효하지 않으면 최소값 사용
                if search_start <= start_idx or search_start >= end_idx:
                    search_start = min(start_idx + min_chunk_size, end_idx - 1)
                    if search_start <= start_idx:
                        search_start = start_idx + 1
                
                # 문장 종료 기호 우선 검색 (다국어 지원)
                # 영어, 한국어, 일본어, 중국어 문장 종료 기호 포함
                sentence_endings = [
                    ". ", "! ", "? ",           # 영어 (공백 포함)
                    ".\n", "!\n", "?\n",        # 영어 (줄바꿈 포함)
                    "。", "！", "？",            # 일본어/중국어 (전각 문자, 공백 없음)
                    "。\n", "！\n", "？\n",      # 일본어/중국어 (줄바꿈 포함)
                    ".", "!", "?"               # 영어 (공백 없음)
                ]
                sentence_end_token_ids = []
                for ending in sentence_endings:
                    ending_tokens = self.tokenizer.encode(ending, add_special_tokens=False)
                    if ending_tokens:
                        sentence_end_token_ids.append((ending, ending_tokens))
                
                # 먼저 문장 종료 기호를 찾기
                found_sentence_end = False
                for ending_text, ending_tokens in sentence_end_token_ids:
                    if not ending_tokens:
                        continue
                    
                    # 구분자 토큰 시퀀스를 찾기 (뒤에서 앞으로)
                    # 인덱스 범위 체크: end_idx - len(ending_tokens) >= 0이어야 함
                    search_end = max(0, end_idx - len(ending_tokens))
                    if search_end < search_start:
                        continue
                    
                    for i in range(search_end, search_start - 1, -1):
                        if i + len(ending_tokens) > total_tokens:
                            continue
                        if tokens[i:i+len(ending_tokens)] == ending_tokens:
                            # 문장 종료 기호 처리
                            next_idx = i + len(ending_tokens)
                            
                            # 일본어/중국어 전각 문자는 공백 없이도 문장 종료로 인정
                            is_fullwidth = ending_text in ['。', '！', '？']
                            
                            if next_idx < total_tokens:
                                if is_fullwidth:
                                    # 전각 문자는 공백 없이도 문장 종료로 인정
                                    best_split_idx = next_idx
                                    found_sentence_end = True
                                    break
                                else:
                                    # 영어 문장 종료 기호는 공백/줄바꿈 확인
                                    next_char_tokens = tokens[next_idx:next_idx+1]
                                    next_text = self.tokenizer.decode(next_char_tokens, skip_special_tokens=True)
                                    # 공백, 줄바꿈, 또는 텍스트 끝이면 문장 종료로 인정
                                    if next_text in [' ', '\n', '\t'] or next_idx >= total_tokens - 1:
                                        best_split_idx = next_idx
                                        found_sentence_end = True
                                        break
                            else:
                                # 텍스트 끝이면 문장 종료로 인정
                                best_split_idx = next_idx
                                found_sentence_end = True
                                break
                    
                    if found_sentence_end:
                        break
                
                # 문장 종료 기호를 찾지 못했으면 일반 구분자 검색
                if not found_sentence_end:
                    for sep_text, sep_tokens in separator_token_ids:
                        if not sep_tokens:
                            continue
                        
                        # 구분자 토큰 시퀀스를 찾기 (뒤에서 앞으로)
                        # 인덱스 범위 체크
                        search_end = max(0, end_idx - len(sep_tokens))
                        if search_end < search_start:
                            continue
                        
                        for i in range(search_end, search_start - 1, -1):
                            if i + len(sep_tokens) > total_tokens:
                                continue
                            if tokens[i:i+len(sep_tokens)] == sep_tokens:
                                best_split_idx = i + len(sep_tokens)
                                break
                        
                        if best_split_idx < end_idx:
                            break
                
                # 최소 청크 크기를 보장 (너무 작은 청크 방지)
                if best_split_idx - start_idx < min_chunk_size:
                    # 최소 크기보다 작으면 원래 end_idx 사용 (강제 분할)
                    best_split_idx = end_idx
                
                # best_split_idx가 start_idx보다 작거나 같으면 안 됨
                if best_split_idx <= start_idx:
                    best_split_idx = end_idx
                
                end_idx = best_split_idx
            
            # 토큰을 텍스트로 디코딩
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunk_text = chunk_text.strip()
            
            # 빈 청크 방지 (하지만 start_idx는 반드시 증가해야 함)
            if chunk_text:
                chunks.append(chunk_text)
            else:
                # 빈 청크가 생성되면 로깅 (이론적으로는 발생하지 않아야 함)
                LOGGER.warning(f"빈 청크 생성됨: start_idx={start_idx}, end_idx={end_idx}")
            
            # 다음 청크 시작 위치 (오버랩 고려)
            # 무한 루프 방지: 최소 1토큰은 전진해야 함
            if chunk_overlap > 0 and end_idx < total_tokens:
                # 오버랩을 고려하되, start_idx는 반드시 증가해야 함
                # 오버랩이 chunk_size보다 크면 제한
                effective_overlap = min(chunk_overlap, chunk_size - 1)
                overlap_start = end_idx - effective_overlap
                start_idx = max(start_idx + 1, overlap_start)
                # start_idx가 end_idx보다 크거나 같으면 안 됨
                if start_idx >= end_idx:
                    start_idx = end_idx
            else:
                start_idx = end_idx
            
            # 무한 루프 방지: start_idx가 total_tokens에 도달하면 종료
            if start_idx >= total_tokens:
                break
            
            # 추가 안전장치: start_idx가 증가하지 않으면 강제로 증가
            if start_idx <= prev_start_idx and start_idx < total_tokens:
                LOGGER.warning(f"start_idx가 증가하지 않음: {start_idx}, 강제로 증가")
                start_idx = min(start_idx + 1, total_tokens)
            
            prev_start_idx = start_idx
        
        return chunks
    
    async def __call__(
        self,
        state: TranslationState,
        config: Optional[Dict[str, Any]] = None
    ) -> TranslationState:
        """LangGraph에서 호출되는 메서드 """
        self.agent_start_time = time.time()
        if self.service_info:
            state["service_info"] = self.service_info
        
        try:
            await self.preprocess(state)
            await self.run(state, config)
        except Exception as e:
            import traceback
            error_msg = f"TranslateAgent 오류: {str(e)}\n{traceback.format_exc()}"
            LOGGER.error(error_msg)
            raise
        
        return state
    
    async def preprocess(self, state: TranslationState) -> TranslationState:
        """전처리 """
        # service_info에서 lang_cd 가져오기 (orchestrator에서 설정됨)
        # lang_cd가 직접 전달되면 사용, 없으면 trans_lang에서 변환
        if self.service_info and "lang_cd" in self.service_info:
            # orchestrator에서 lang_cd를 직접 설정한 경우
            trnsl_lang_cd = self.service_info.get("lang_cd", "en")
        else:
            # trans_lang을 사용하는 경우 (기존 로직)
            TRNSL_LANG_CD_MAP = {
                "ENG": "en",
                "KOR": "ko",
                "JPN": "ja",
                "CHN": "zh",
                "VNM": "vi",
                "FRA": "fr",
                "THA": "th",
                "PHL": "tl",
                "KHM": "km",
            }
            trnsl_lang_cd = TRNSL_LANG_CD_MAP.get(
                self.service_info.get("trans_lang", "ENG") if self.service_info else "ENG",
                "en"
            )
        
        # generate_determine_prompt를 사용하여 프롬프트 생성
        target_lang_name = get_language_name(trnsl_lang_cd)
        prompt = create_translation_prompt(trnsl_lang_cd, target_lang_name)
        
        # 프롬프트 토큰 수 로깅 
        # TODO: 실제 토큰 수 계산 로직 추가 필요
        token = len(prompt.split())  # 간단한 단어 수로 대체
        LOGGER.info(f"프롬프트 토큰 수: {token}")
        
        # state에 프롬프트 저장
        state["prompt"] = prompt
        state["target_lang_cd"] = trnsl_lang_cd
        state["target_lang_name"] = target_lang_name
        
        return state
    
    async def run(
        self,
        state: TranslationState,
        config: Optional[Dict[str, Any]] = None
    ) -> TranslationState:
        """
        사용자의 질문을 LLM을 통해 검색에 최적화된 쿼리로 재작성합니다.
        청크 분할 및 병렬 처리를 포함합니다.
        """
        # 1. 프롬프트는 preprocess()에서 이미 생성됨
        base_prompt = state["prompt"]
        
        # 2. 청크 나누기 (토큰 수 기반으로 분할)
        original_text = state["original_text"]
        original_text_length = len(original_text)
        
        # 토큰 수를 계산하는 함수
        def count_tokens(text: str) -> int:
            """텍스트의 토큰 수를 반환"""
            if self.tokenizer is not None:
                # transformers 토크나이저 사용
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            else:
                # 폴백: 간단한 근사치 (공백 기준 단어 수 * 1.3)
                # 정확하지 않지만 기본 동작은 유지
                return int(len(text.split()) * 1.3)
        
        original_token_count = count_tokens(original_text)
        
        log_msg = f"[청크 분할] 원본 텍스트: {original_text_length}자, {original_token_count}토큰"
        LOGGER.info(log_msg)
        print(log_msg)  # 터미널 출력 보장
        
        if original_token_count <= self.chunk_size:
            chunks = [original_text]
            log_msg = f"[청크 분할] 텍스트가 작아서 청크 분할하지 않음 ({original_token_count}토큰 <= {self.chunk_size}토큰)"
            LOGGER.info(log_msg)
            print(log_msg)  # 터미널 출력 보장
        else:
            # 토큰 수 기반으로 청크 분할 (정확한 토큰 수 기준)
            log_msg = f"[청크 분할] 청크 크기: {self.chunk_size}토큰, 오버랩: {self.chunk_overlap}토큰"
            LOGGER.info(log_msg)
            print(log_msg)  # 터미널 출력 보장
            
            # 토큰 수 기반으로 정확하게 청크를 나누기
            chunks = self._split_text_by_tokens(
                original_text, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            log_msg = f"[청크 분할] 총 {len(chunks)}개 청크로 분할됨"
            LOGGER.info(log_msg)
            print(log_msg)  # 터미널 출력 보장
            
            # 각 청크의 토큰 수 로깅
            for i, chunk in enumerate(chunks):
                chunk_tokens = count_tokens(chunk)
                LOGGER.info(f"[청크 {i}] 토큰 수: {chunk_tokens}토큰, 문자 수: {len(chunk)}자")
        
        # config에서 Queue와 CustomAsyncCallbackHandler 클래스 가져오기 (간소화)
        run_config = config or {}
        
        # callbacks에서 queue 추출
        callbacks = run_config.get("callbacks", [])
        if hasattr(callbacks, 'handlers'):
            callbacks_list = callbacks.handlers
        elif isinstance(callbacks, list):
            callbacks_list = callbacks
        else:
            callbacks_list = [callbacks] if callbacks else []
        
        queue = None
        if callbacks_list and hasattr(callbacks_list[0], 'queue'):
            queue = callbacks_list[0].queue
        
        # configurable에서 CustomAsyncCallbackHandler 추출
        configurable = run_config.get("configurable", {})
        CustomAsyncCallbackHandler = configurable.get("callback_handler_class")
        
        if not queue:
            raise ValueError("Queue가 필요합니다. 스트리밍 방식으로만 동작합니다.")
        if not CustomAsyncCallbackHandler:
            raise ValueError("CustomAsyncCallbackHandler 클래스가 필요합니다.")
        
        # llm을 외부 스코프에서 캡처 (중첩 함수에서 self 접근 문제 해결)
        llm_instance = self.llm
        
        # 공통 번역 함수 (astream 최적화 + 인덱스 기반 callback)
        async def translate_single_chunk(chunk: str, prompt: str, chunk_index: int) -> str:
            """단일 청크를 번역하는 공통 함수 (인덱스 기반 callback)"""
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=chunk)
            ]
            
            # 인덱스별 callback 생성 (이미지 코드 구조 - queue만 전달)
            class IndexedQueue:
                """인덱스 정보를 포함한 Queue 래퍼"""
                def __init__(self, queue: Queue, chunk_index: int):
                    self.queue = queue
                    self.chunk_index = chunk_index
                
                async def put(self, item):
                    """인덱스와 함께 Queue에 넣기"""
                    await self.queue.put((self.chunk_index, item))
            
            indexed_queue = IndexedQueue(queue, chunk_index)
            callback_instance = CustomAsyncCallbackHandler(indexed_queue)
            callbacks = [callback_instance]
            
            # astream을 사용하여 스트리밍 번역 (각 토큰을 즉시 Queue에 넣기)
            from langchain_core.runnables.config import RunnableConfig
            result = ""
            last_chunk_response = None
            
            run_config = RunnableConfig(callbacks=callbacks)
            async for chunk_response in llm_instance.astream(messages, config=run_config):
                # 콘텐츠가 있으면 바로 누적하고 Queue에 즉시 전송
                if chunk_response.content:
                    # 토큰을 그대로 Queue에 넣기 (한 글자씩 쪼개지 않음)
                    await indexed_queue.put(chunk_response.content)
                    result += chunk_response.content
                
                # 마지막 청크에서만 토큰 사용량 추적 (성능 최적화)
                last_chunk_response = chunk_response
            
            # 토큰 사용량 추적 (마지막 청크에서만, 이미지 코드와 동일한 구조)
            if last_chunk_response and last_chunk_response.response_metadata:
                usage = last_chunk_response.response_metadata.get("usage", {})
                if usage:
                    self.crt_step_inpt_tokn_cnt = usage.get("prompt_tokens", 0)
                    self.crt_step_otpt_tokn_cnt = usage.get("completion_tokens", 0)
                    self.crt_step_totl_tokn_cnt = usage.get("total_tokens", 0)
                else:
                    # usage가 없으면 기본값 0으로 설정
                    self.crt_step_inpt_tokn_cnt = 0
                    self.crt_step_otpt_tokn_cnt = 0
                    self.crt_step_totl_tokn_cnt = 0
            
            # 텍스트 교체 로직 (이미지 코드와 동일, 최적화)
            if '\n\n' in result:
                result = result.replace('\n\n', "<<>>")
                result = result.replace('\n', '\n\n')
                result = result.replace('<<>>', '\n\n')
            elif '\n' in result:
                result = result.replace('\n', '\n\n')
            
            return result.strip() if result else ""
        
        # 청크가 1개면 기존 방식대로 처리 (병렬 처리 오버헤드 제거)
        if len(chunks) == 1:
            result = await translate_single_chunk(chunks[0], base_prompt, 0)  # 인덱스 0부터 시작
            state['answer'] = result
            return state
        
        # 3. 청크가 여러 개일 때 병렬 처리 (인덱스 0부터 시작)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def translate_chunk_parallel(
            chunk: str,
            chunk_index: int,  # 0부터 시작하는 인덱스
            previous_chunk: Optional[str] = None,
            next_chunk: Optional[str] = None
        ) -> tuple[int, str]:
            """세마포어를 사용한 병렬 번역 (인덱스 기반)"""
            async with semaphore:
                # 문맥 정보를 포함한 프롬프트 생성
                context_info = []
                if previous_chunk:
                    prev_context = previous_chunk[-150:] if len(previous_chunk) > 150 else previous_chunk
                    context_info.append(f"\n[이전 문맥 (참고용)]:\n{prev_context}")
                if next_chunk:
                    next_context = next_chunk[:150] if len(next_chunk) > 150 else next_chunk
                    context_info.append(f"\n[다음 문맥 (참고용)]:\n{next_context}")
                
                if context_info:
                    context_section = "\n".join(context_info)
                    context_instruction = "\n\n중요: 위의 이전/다음 문맥 정보를 참고하여 자연스럽고 일관된 번역을 제공하세요. 하지만 반드시 주어진 텍스트만 정확하게 번역하세요."
                    context_prompt = f"{base_prompt}{context_section}{context_instruction}"
                else:
                    context_prompt = base_prompt
                
                # 공통 번역 함수 사용 (인덱스 전달)
                result = await translate_single_chunk(chunk, context_prompt, chunk_index)
                return chunk_index, result
        
        # 첫 번째 청크를 먼저 시작하고, 첫 토큰 생성 시 나머지 청크 병렬 시작
        first_token_received = asyncio.Event()
        first_token_time = None  # 첫 토큰 시간 측정
        chunk_start_time = time.time()  # 청크 0 시작 시간
        
        # 첫 번째 청크 전용 번역 함수 (첫 토큰 감지)
        async def translate_first_chunk():
            """첫 번째 청크를 번역하고 첫 토큰 수신 시 이벤트 발생"""
            nonlocal first_token_time
            chunk = chunks[0]
            chunk_index = 0
            next_chunk = chunks[1] if len(chunks) > 1 else None
            result = ""  # 함수 시작 부분에서 초기화
            
            async with semaphore:
                # 문맥 정보 생성
                context_info = []
                if next_chunk:
                    next_context = next_chunk[:150] if len(next_chunk) > 150 else next_chunk
                    context_info.append(f"\n[다음 문맥 (참고용)]:\n{next_context}")
                
                if context_info:
                    context_section = "\n".join(context_info)
                    context_instruction = "\n\n중요: 위의 이전/다음 문맥 정보를 참고하여 자연스럽고 일관된 번역을 제공하세요. 하지만 반드시 주어진 텍스트만 정확하게 번역하세요."
                    context_prompt = f"{base_prompt}{context_section}{context_instruction}"
                else:
                    context_prompt = base_prompt
                
                # 번역 시작
                messages = [
                    SystemMessage(content=context_prompt),
                    HumanMessage(content=chunk)
                ]
                
                class IndexedQueue:
                    def __init__(self, queue: Queue, chunk_index: int):
                        self.queue = queue
                        self.chunk_index = chunk_index
                        self.first_token_sent = False
                    
                    async def put(self, item):
                        nonlocal first_token_time
                        # 첫 토큰 수신 시 이벤트 발생 및 시간 측정
                        if not self.first_token_sent and item is not None:
                            self.first_token_sent = True
                            first_token_time = time.time()
                            elapsed = first_token_time - chunk_start_time
                            first_token_received.set()
                            LOGGER.info(f"⚡ [첫 토큰 생성] {elapsed:.3f}초 소요 (청크 0)")
                            print(f"⚡ [첫 토큰 생성] {elapsed:.3f}초 소요 (청크 0)")
                            LOGGER.info(f"[병렬 처리 트리거] 나머지 청크 {len(chunks)-1}개 시작")
                            print(f"[병렬 처리 트리거] 나머지 청크 {len(chunks)-1}개 시작")
                        await self.queue.put((self.chunk_index, item))
                
                indexed_queue = IndexedQueue(queue, chunk_index)
                callback_instance = CustomAsyncCallbackHandler(indexed_queue)
                callbacks = [callback_instance]
                
                from langchain_core.runnables.config import RunnableConfig
                last_chunk_response = None
                
                run_config = RunnableConfig(callbacks=callbacks)
                async for chunk_response in llm_instance.astream(messages, config=run_config):
                    if chunk_response.content:
                        # 토큰을 그대로 Queue에 넣기 (한 글자씩 쪼개지 않음)
                        await indexed_queue.put(chunk_response.content)
                        result += chunk_response.content
                    last_chunk_response = chunk_response
                
                # 토큰 사용량 추적
                if last_chunk_response and last_chunk_response.response_metadata:
                    usage = last_chunk_response.response_metadata.get("usage", {})
                    if usage:
                        self.crt_step_inpt_tokn_cnt = usage.get("prompt_tokens", 0)
                        self.crt_step_otpt_tokn_cnt = usage.get("completion_tokens", 0)
                        self.crt_step_totl_tokn_cnt = usage.get("total_tokens", 0)
                
                # 텍스트 교체 로직
                if '\n\n' in result:
                    result = result.replace('\n\n', "<<>>")
                    result = result.replace('\n', '\n\n')
                    result = result.replace('<<>>', '\n\n')
                elif '\n' in result:
                    result = result.replace('\n', '\n\n')
                
                return chunk_index, result.strip() if result else ""
        
        # 나머지 청크들을 병렬로 처리하는 함수
        async def translate_remaining_chunks():
            """첫 토큰 수신 후 나머지 청크들을 병렬 처리"""
            # 외부에서 이미 첫 토큰을 기다렸으므로 여기서는 바로 시작
            
            # 나머지 청크가 없으면 빈 리스트 반환
            if len(chunks) <= 1:
                return []
            
            LOGGER.info(f"[병렬 처리 시작] 청크 1~{len(chunks)-1} 병렬 번역 시작")
            print(f"[병렬 처리 시작] 청크 1~{len(chunks)-1} 병렬 번역 시작")
            
            # 나머지 청크 Task 생성
            tasks = []
            for i in range(1, len(chunks)):
                previous_chunk = chunks[i - 1]
                next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None
                tasks.append(translate_chunk_parallel(chunks[i], i, previous_chunk, next_chunk))
            
            # 병렬 실행 (결과를 리스트로 명시적 변환)
            # return_exceptions=True: 일부 청크가 실패해도 나머지는 계속 진행
            if tasks:
                results = await asyncio.gather(*tasks)
                return list(results)
            else:
                return []
        
        # 첫 번째 청크를 먼저 시작 (스트리밍은 즉시 시작됨)
        first_chunk_task = asyncio.create_task(translate_first_chunk())
        
        # 첫 토큰이 나올 때까지 기다린 후 나머지 청크들을 시작
        # 첫 토큰이 나오면 first_token_received 이벤트가 설정됨
        await first_token_received.wait()
        
        LOGGER.info(f"[첫 토큰 수신 완료] 나머지 청크 {len(chunks)-1}개 시작")
        print(f"[첫 토큰 수신 완료] 나머지 청크 {len(chunks)-1}개 시작")
        
        # 첫 토큰 이후에 나머지 청크들을 시작
        remaining_chunks_task = asyncio.create_task(translate_remaining_chunks())
        
        # 모든 청크를 백그라운드에서 실행하고, 완료될 때까지 대기
        # 첫 토큰은 이미 Queue를 통해 스트리밍되고 있음
        first_result, remaining_results = await asyncio.gather(
            first_chunk_task,
            remaining_chunks_task
        )
        
        # 결과 합치기 (remaining_results를 명시적으로 리스트로 변환)
        if remaining_results:
            # remaining_results가 튜플일 수 있으므로 리스트로 변환
            remaining_list = list(remaining_results) if not isinstance(remaining_results, list) else remaining_results
            results = [first_result] + remaining_list
        else:
            results = [first_result]
        
        # 모든 결과가 튜플 형태인지 확인하고 정렬
        # first_result와 remaining_list의 항목들이 모두 (index, result) 튜플 형태여야 함
        def get_index(item):
            """항목에서 인덱스 추출 (튜플이면 첫 번째 요소, 아니면 0)"""
            if isinstance(item, tuple) and len(item) >= 2:
                return item[0] if isinstance(item[0], int) else 0
            return 0
        
        # 인덱스 순서대로 정렬하여 합치기
        sorted_results = sorted(results, key=get_index)
        
        # 정렬된 결과에서 번역된 텍스트만 추출
        translated_chunks = []
        failed_chunk_indices = []
        for item in sorted_results:
            if isinstance(item, tuple) and len(item) >= 2:
                # (index, translated_text) 튜플 형태
                translated_chunks.append(item[1])
            elif isinstance(item, str):
                # 문자열만 있는 경우
                translated_chunks.append(item)
            else:
                # 기타 경우는 무시
                continue
        
        # 번역된 청크들을 합치기 (chunk_overlap=0이므로 중복 제거 불필요)
        if not translated_chunks:
            final_result = ""
        elif len(translated_chunks) == 1:
            final_result = translated_chunks[0]
        else:
            # 단순히 공백으로 합치기
            merged = " ".join(translated_chunks)
            # 중복된 공백 제거
            final_result = re.sub(r'\s+', ' ', merged).strip()
        
        state['answer'] = final_result
        
        return state
