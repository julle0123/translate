"""번역 노드 - LangGraph 노드에서 사용하는 번역 함수 (이미지 코드 구조)"""
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
from schema.schemas import ServiceInfo, UserInfo

# 로거 설정
LOGGER = logging.getLogger(__name__)


class TranslateAgent:
    """번역 에이전트 클래스 (이미지 코드 구조)"""
    
    def __init__(
        self,
        service_info: Optional[Dict[str, Any]] = None,
        chat_req: Optional[Dict[str, Any]] = None,
        llm: Optional[ChatOpenAI] = None,
        callbacks: Optional[List] = None
    ):
        """
        초기화 (이미지 코드 구조)
        
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
        self.max_concurrent = int(os.getenv("TRANSLATE_MAX_CONCURRENT", "5"))
        
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
        """LangGraph에서 호출되는 메서드 (이미지 코드 구조)"""
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
        """전처리 (이미지 코드 구조)"""
        # TRNSL_LANG_CD_MAP을 사용하여 번역 언어 코드 가져오기
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
        from prompt.prompts import create_translation_prompt
        from utils.language_utils import get_language_name
        
        target_lang_name = get_language_name(trnsl_lang_cd)
        prompt = create_translation_prompt(trnsl_lang_cd, target_lang_name)
        
        # 프롬프트 토큰 수 로깅 (이미지 코드 구조)
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
        # 1. 프롬프트 생성
        base_prompt = create_translation_prompt(
            state["target_lang_cd"],
            state["target_lang_name"]
        )
        
        # 2. 청크 나누기 (동적 크기 조정으로 청크 수 최소화)
        original_text = state["original_text"]
        if len(original_text) <= self.chunk_size:
            chunks = [original_text]
        else:
            # 더 큰 청크 크기 사용 (청크 수 감소로 전체 처리 시간 단축)
            dynamic_chunk_size = min(self.chunk_size * 2, len(original_text))
            
            # 오버랩 없이 청크를 나누기 (중복 번역 방지)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=dynamic_chunk_size,
                chunk_overlap=0,  # 오버랩 제거로 중복 번역 방지
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(original_text)
        
        # config에서 Queue와 CustomAsyncCallbackHandler 클래스 가져오기 (orchestrator에서 전달, 필수)
        run_config = config or {}
        queue = None
        CustomAsyncCallbackHandler = None
        
        # Queue와 CallbackHandler 클래스 추출
        if isinstance(run_config, dict):
            callbacks = run_config.get("callbacks", [])
            if callbacks:
                if hasattr(callbacks, 'handlers'):
                    callbacks_list = callbacks.handlers
                elif isinstance(callbacks, list):
                    callbacks_list = callbacks
                else:
                    callbacks_list = [callbacks] if callbacks else []
                
                if callbacks_list:
                    first_callback = callbacks_list[0]
                    if hasattr(first_callback, 'queue'):
                        queue = first_callback.queue
            
            configurable = run_config.get("configurable", {})
            if not queue:
                queue = configurable.get("streaming_queue")
            CustomAsyncCallbackHandler = configurable.get("callback_handler_class")
        else:
            if hasattr(run_config, "callbacks") and run_config.callbacks:
                callbacks_obj = run_config.callbacks
                if hasattr(callbacks_obj, 'handlers'):
                    callbacks_list = callbacks_obj.handlers
                elif isinstance(callbacks_obj, list):
                    callbacks_list = callbacks_obj
                else:
                    callbacks_list = [callbacks_obj] if callbacks_obj else []
                
                if callbacks_list:
                    first_callback = callbacks_list[0]
                    if hasattr(first_callback, 'queue'):
                        queue = first_callback.queue
            
            if hasattr(run_config, "configurable") and run_config.configurable:
                if not queue:
                    queue = run_config.configurable.get("streaming_queue")
                CustomAsyncCallbackHandler = run_config.configurable.get("callback_handler_class")
        
        # Queue와 CustomAsyncCallbackHandler가 없으면 에러 (무조건 스트리밍 방식 사용)
        if not queue:
            raise ValueError("Queue가 필요합니다. 스트리밍 방식으로만 동작합니다.")
        if not CustomAsyncCallbackHandler:
            raise ValueError("CustomAsyncCallbackHandler 클래스가 필요합니다. orchestrator에서 전달해야 합니다.")
        
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
            
            # astream을 사용하여 스트리밍 번역 (callbacks 직접 전달)
            result = ""
            last_chunk_response = None
            
            async for chunk_response in llm_instance.astream(messages, callbacks=callbacks):
                # 콘텐츠가 있으면 바로 누적
                if chunk_response.content:
                    result += chunk_response.content
                
                # 마지막 청크에서만 토큰 사용량 추적 (성능 최적화)
                last_chunk_response = chunk_response
            
            # 토큰 사용량 추적 (마지막 청크에서만, 이미지 코드와 동일한 구조)
            if last_chunk_response and last_chunk_response.response_metadata:
                self.crt_step_inpt_tokn_cnt = last_chunk_response.response_metadata["usage"]["prompt_tokens"]
                self.crt_step_otpt_tokn_cnt = last_chunk_response.response_metadata["usage"]["completion_tokens"]
                self.crt_step_totl_tokn_cnt = last_chunk_response.response_metadata["usage"]["total_tokens"]
            
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
        
        # 모든 청크를 병렬로 번역 (인덱스는 0부터 시작)
        tasks = []
        for i, chunk in enumerate(chunks, start=0):  # 0부터 시작
            previous_chunk = chunks[i - 2] if i > 1 else None
            next_chunk = chunks[i] if i < len(chunks) else None
            tasks.append(translate_chunk_parallel(chunk, i, previous_chunk, next_chunk))
        
        results = await asyncio.gather(*tasks)
        
        # 인덱스 순서대로 정렬하여 합치기
        sorted_results = sorted(results, key=lambda x: x[0])
        translated_chunks = [translated for _, translated in sorted_results]
        
        # 재작성된 쿼리를 상태에 업데이트 (이미지 코드와 동일)
        if not translated_chunks:
            final_result = ""
        elif len(translated_chunks) == 1:
            final_result = translated_chunks[0]
        else:
            # 중복 제거: 각 청크가 이전 청크의 끝 부분과 겹치는지 확인 (최적화)
            deduplicated_chunks = [translated_chunks[0]]
            for i in range(1, len(translated_chunks)):
                chunk = translated_chunks[i]
                prev_chunk = deduplicated_chunks[-1]
                
                # 빠른 중복 체크: 이전 청크의 마지막 50자와 현재 청크의 처음 50자만 비교
                check_len = min(50, len(prev_chunk), len(chunk))
                if check_len > 0:
                    prev_end = prev_chunk[-check_len:]
                    curr_start = chunk[:check_len]
                    if prev_end == curr_start:
                        # 겹치는 부분 제거
                        deduplicated_chunks.append(chunk[check_len:])
                    else:
                        # 더 긴 겹침 확인 (최대 100자)
                        overlap_found = False
                        for overlap_len in range(min(100, len(prev_chunk), len(chunk)), check_len, -1):
                            if prev_chunk[-overlap_len:] == chunk[:overlap_len]:
                                deduplicated_chunks.append(chunk[overlap_len:])
                                overlap_found = True
                                break
                        if not overlap_found:
                            deduplicated_chunks.append(chunk)
                else:
                    deduplicated_chunks.append(chunk)
            
            # 중복 제거된 청크들을 합치기
            merged = " ".join(deduplicated_chunks)
            # 중복된 공백 제거
            merged = re.sub(r'\s+', ' ', merged)
            final_result = merged.strip()
        
        state['answer'] = final_result
        
        return state
