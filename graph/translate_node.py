"""번역 노드 - LangGraph 노드에서 사용하는 번역 함수"""
import os
import asyncio
import re
import logging
from typing import Dict, Any, List, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage

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
        state: TranslationState
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
        
        # 2. 청크 나누기
        original_text = state["original_text"]
        if len(original_text) <= self.chunk_size:
            chunks = [original_text]
        else:
            # 오버랩 없이 청크를 나누기 (중복 번역 방지)
            # 문맥 정보는 이전/다음 청크의 일부를 프롬프트에 포함하므로 오버랩 불필요
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=0,  # 오버랩 제거로 중복 번역 방지
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(original_text)
        
        # 공통 번역 함수 (astream + 토큰 추적 + 텍스트 교체)
        async def translate_single_chunk(chunk: str, prompt: str) -> str:
            """단일 청크를 번역하는 공통 함수"""
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=chunk)
            ]
            
            # astream을 사용하여 스트리밍 번역
            result = ""
            async for chunk_response in self.llm.astream(messages):
                result += chunk_response.content
                
                # 토큰 사용량 추적 (이미지 코드와 동일한 구조)
                if hasattr(chunk_response, 'response_metadata') and chunk_response.response_metadata:
                    self.crt_step_inpt_tokn_cnt = chunk_response.response_metadata["usage"]["prompt_tokens"]
                    self.crt_step_otpt_tokn_cnt = chunk_response.response_metadata["usage"]["completion_tokens"]
                    self.crt_step_totl_tokn_cnt = chunk_response.response_metadata["usage"]["total_tokens"]
            
            # 텍스트 교체 로직 (이미지 코드와 동일)
            result = result.replace('\n\n', "<<>>")
            result = result.replace('\n', '\n\n')
            result = result.replace('<<>>', '\n\n')
            
            return result.strip() if result else ""
        
        # 청크가 1개면 기존 방식대로 처리 (병렬 처리 오버헤드 제거)
        if len(chunks) == 1:
            result = await translate_single_chunk(chunks[0], base_prompt)
            state['answer'] = result
            return state
        
        # 3. 청크가 여러 개일 때만 병렬 처리
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def translate_chunk_parallel(
            chunk: str,
            index: int,
            previous_chunk: Optional[str] = None,
            next_chunk: Optional[str] = None
        ) -> tuple[int, str]:
            """세마포어를 사용한 병렬 번역"""
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
                
                # 공통 번역 함수 사용
                result = await translate_single_chunk(chunk, context_prompt)
                return index, result
        
        # 모든 청크를 병렬로 번역
        tasks = []
        for i, chunk in enumerate(chunks):
            previous_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None
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


# LangGraph 노드에서 직접 사용할 수 있는 함수
async def translate_node(state: TranslationState) -> Dict[str, Any]:
    """
    번역 노드 함수 - LangGraph에서 직접 사용
    
    Args:
        state: 번역 상태 (LLM은 내부에서 생성)
    
    Returns:
        dict: 번역 결과
    """
    # LLM 생성
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        streaming=True,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # TranslateNode 인스턴스 생성 및 실행
    node = TranslateNode(llm=llm)
    return await node.run(state)
