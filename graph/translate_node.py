"""번역 노드 - LangGraph 노드에서 사용하는 번역 함수"""
import os
import asyncio
import re
import logging
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


class TranslateNode:
    """번역 노드 클래스 - 프롬프트 호출, 청크 분할, 병렬 astream 처리"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        callback: Optional[Callable] = None,
        service_info: Optional[ServiceInfo] = None,
        user_info: Optional[UserInfo] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_concurrent: Optional[int] = None
    ):
        """
        초기화
        
        Args:
            llm: LLM 인스턴스
            callback: 콜백 함수 (선택사항)
            service_info: 서비스 정보 (선택사항)
            user_info: 사용자 정보 (선택사항)
            chunk_size: 청크 크기 (기본값: 환경변수 또는 2000)
            chunk_overlap: 청크 오버랩 (기본값: 환경변수 또는 200)
            max_concurrent: 최대 동시 처리 수 (기본값: 환경변수 또는 5)
        """
        self.llm = llm
        self.callback = callback
        self.service_info = service_info
        self.user_info = user_info
        
        # 설정값 (환경변수 우선, 없으면 기본값)
        self.chunk_size = chunk_size or int(os.getenv("TRANSLATE_CHUNK_SIZE", "2000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("TRANSLATE_CHUNK_OVERLAP", "200"))
        self.max_concurrent = max_concurrent or int(os.getenv("TRANSLATE_MAX_CONCURRENT", "5"))
        
        # 토큰 사용량 추적 (이미지 코드와 동일한 변수명)
        self.crt_step_inpt_tokn_cnt = 0
        self.crt_step_otpt_tokn_cnt = 0
        self.crt_step_totl_tokn_cnt = 0
        
        LOGGER.info(f"TranslateNode 초기화: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, max_concurrent={self.max_concurrent}")
    
    async def run(
        self,
        state: TranslationState,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            # 텍스트가 길수록 더 큰 청크 사용
            dynamic_chunk_size = min(self.chunk_size * 2, len(original_text))
            
            # 오버랩 없이 청크를 나누기 (중복 번역 방지)
            # 문맥 정보는 이전/다음 청크의 일부를 프롬프트에 포함하므로 오버랩 불필요
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
        # callbacks에서 Queue 추출
        if isinstance(run_config, dict):
            callbacks = run_config.get("callbacks", [])
            if callbacks:
                # AsyncCallbackManager인 경우 handlers 속성 사용
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
            
            # configurable에서 추출
            configurable = run_config.get("configurable", {})
            if not queue:
                queue = configurable.get("streaming_queue")
            CustomAsyncCallbackHandler = configurable.get("callback_handler_class")
        else:
            # RunnableConfig 객체인 경우
            if hasattr(run_config, "callbacks") and run_config.callbacks:
                callbacks_obj = run_config.callbacks
                # AsyncCallbackManager인 경우 handlers 속성 사용
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
        
        # 공통 번역 함수 (astream 최적화 + 인덱스 기반 callback)
        async def translate_single_chunk(chunk: str, prompt: str, chunk_index: int) -> str:
            """단일 청크를 번역하는 공통 함수 (인덱스 기반 callback)"""
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=chunk)
            ]
            
            # 인덱스별 callback 생성 (이미지 코드 구조 - queue만 전달)
            # CustomAsyncCallbackHandler는 이미지 코드 그대로 유지하고, 
            # 인덱스 정보를 포함한 래퍼 Queue 사용
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
            chunk_config = {"callbacks": callbacks}
            
            # astream을 사용하여 스트리밍 번역 (config 포함)
            result = ""
            last_chunk_response = None
            
            async for chunk_response in self.llm.astream(messages, config=chunk_config):
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
            result = await translate_single_chunk(chunks[0], base_prompt, 1)  # 인덱스 1부터 시작
            state['answer'] = result
            return state
        
        # 3. 청크가 여러 개일 때 병렬 처리 (인덱스 1부터 시작)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def translate_chunk_parallel(
            chunk: str,
            chunk_index: int,  # 1부터 시작하는 인덱스
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
        
        # 모든 청크를 병렬로 번역 (인덱스는 1부터 시작)
        tasks = []
        for i, chunk in enumerate(chunks, start=1):  # 1부터 시작
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

    async def __call__(
        self,
        state: TranslationState,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """LangGraph에서 노드처럼 호출될 때 run을 실행"""
        return await self.run(state, config)


# LangGraph 노드에서 직접 사용할 수 있는 함수
async def translate_node(state: TranslationState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    번역 노드 함수 - LangGraph에서 직접 사용
    
    Args:
        state: 번역 상태 (LLM은 내부에서 생성)
        config: LangGraph에서 전달하는 config (callbacks 포함)
    
    Returns:
        dict: 번역 결과
    """
    from langchain_core.runnables import RunnableConfig
    
    # LLM 생성
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        streaming=True,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # TranslateNode 인스턴스 생성 및 실행 (config 전달)
    node = TranslateNode(llm=llm)
    return await node(state, config)
