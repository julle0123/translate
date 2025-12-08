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
        self.chunk_size = int(os.getenv("TRANSLATE_CHUNK_SIZE", "2000"))
        self.chunk_overlap = int(os.getenv("TRANSLATE_CHUNK_OVERLAP", "200"))
        self.max_concurrent = int(os.getenv("TRANSLATE_MAX_CONCURRENT", "3"))
        
        # 토큰 사용량 추적 (이미지 코드와 동일한 변수명)
        self.crt_step_inpt_tokn_cnt = 0
        self.crt_step_otpt_tokn_cnt = 0
        self.crt_step_totl_tokn_cnt = 0
        
        LOGGER.info(f"TranslateAgent 초기화: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, max_concurrent={self.max_concurrent}")
    
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
        
        # 2. 청크 나누기 (동적 크기 조정으로 청크 수 최소화)
        original_text = state["original_text"]
        original_text_length = len(original_text)
        log_msg = f"[청크 분할] 원본 텍스트 길이: {original_text_length}자"
        LOGGER.info(log_msg)
        print(log_msg)  # 터미널 출력 보장
        
        if original_text_length <= self.chunk_size:
            chunks = [original_text]
            log_msg = f"[청크 분할] 텍스트가 작아서 청크 분할하지 않음 (길이: {original_text_length} <= {self.chunk_size})"
            LOGGER.info(log_msg)
            print(log_msg)  # 터미널 출력 보장
        else:
            # 더 큰 청크 크기 사용 (청크 수 감소로 전체 처리 시간 단축)
            dynamic_chunk_size = 1000
            log_msg = f"[청크 분할] 동적 청크 크기: {dynamic_chunk_size}자 (기본 청크 크기: {self.chunk_size}자)"
            LOGGER.info(log_msg)
            print(log_msg)  # 터미널 출력 보장
            
            # 오버랩 없이 청크를 나누기 (중복 번역 방지)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=dynamic_chunk_size,
                chunk_overlap=0,  # 오버랩 제거로 중복 번역 방지
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(original_text)
            log_msg = f"[청크 분할] 총 {len(chunks)}개 청크로 분할됨"
            LOGGER.info(log_msg)
            print(log_msg)  # 터미널 출력 보장
        
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
            
            # astream을 사용하여 스트리밍 번역 (config로 callbacks 전달)
            from langchain_core.runnables.config import RunnableConfig
            result = ""
            last_chunk_response = None
            
            run_config = RunnableConfig(callbacks=callbacks)
            async for chunk_response in llm_instance.astream(messages, config=run_config):
                # 콘텐츠가 있으면 바로 누적
                if chunk_response.content:
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
        
        # 첫 번째 청크(청크 0)를 먼저 시작하여 첫 토큰 응답 시간 개선
        tasks = []
        
        # 청크 0 Task를 먼저 생성하여 OpenAI 요청을 먼저 보냄
        next_chunk_for_0 = chunks[1] if len(chunks) > 1 else None
        task_0 = translate_chunk_parallel(chunks[0], 0, None, next_chunk_for_0)
        tasks.append(task_0)
        
        # 나머지 청크 Task 생성 (청크 1부터)
        for i in range(1, len(chunks)):
            previous_chunk = chunks[i - 1]
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None
            tasks.append(translate_chunk_parallel(chunks[i], i, previous_chunk, next_chunk))
        
        # 모든 Task를 동시에 실행 (청크 0이 먼저 시작되었으므로 첫 토큰이 빨리 도착)
        results = await asyncio.gather(*tasks)
        
        # 인덱스 순서대로 정렬하여 합치기
        sorted_results = sorted(results, key=lambda x: x[0])
        translated_chunks = [translated for _, translated in sorted_results]
        
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
