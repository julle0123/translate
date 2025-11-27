"""번역 생성기 구현 클래스"""
import asyncio
import re
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage

from generator.base_generator import BaseGenerator
from prompt.prompts import create_translation_prompt


class TranslationGenerator(BaseGenerator):
    """번역 생성기 구현 클래스"""
    
    def split_text(self, text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
        """텍스트를 청크로 분할"""
        if len(text) <= chunk_size:
            return [text]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks
    
    async def _translate_chunk(
        self, 
        chunk: str, 
        prompt: str,
        previous_chunk: str = None,
        next_chunk: str = None,
        is_first: bool = False,
        is_last: bool = False
    ) -> str:
        """
        단일 청크를 번역 - 문맥을 고려한 번역
        
        Args:
            chunk: 번역할 청크
            prompt: 기본 프롬프트
            previous_chunk: 이전 청크 (문맥 제공)
            next_chunk: 다음 청크 (문맥 제공)
            is_first: 첫 번째 청크 여부
            is_last: 마지막 청크 여부
        """
        # 문맥 정보를 포함한 프롬프트 생성
        context_prompt = self._build_contextual_prompt(
            prompt, 
            previous_chunk, 
            next_chunk, 
            is_first, 
            is_last
        )
        
        messages = [
            SystemMessage(content=context_prompt),
            HumanMessage(content=chunk)
        ]
        
        translated_text = ""
        async for chunk_response in self.llm.astream(messages):
            content = chunk_response.content if hasattr(chunk_response, 'content') else str(chunk_response)
            if content:
                translated_text += content
        
        return translated_text.strip() if translated_text else ""
    
    def _build_contextual_prompt(
        self,
        base_prompt: str,
        previous_chunk: str = None,
        next_chunk: str = None,
        is_first: bool = False,
        is_last: bool = False
    ) -> str:
        """문맥 정보를 포함한 프롬프트 생성"""
        context_info = []
        
        if previous_chunk:
            # 이전 청크의 끝 부분만 제공 (약 100자)
            prev_context = previous_chunk[-150:] if len(previous_chunk) > 150 else previous_chunk
            context_info.append(f"\n[이전 문맥 (참고용)]:\n{prev_context}")
        
        if next_chunk:
            # 다음 청크의 시작 부분만 제공 (약 100자)
            next_context = next_chunk[:150] if len(next_chunk) > 150 else next_chunk
            context_info.append(f"\n[다음 문맥 (참고용)]:\n{next_context}")
        
        if context_info:
            context_section = "\n".join(context_info)
            context_instruction = "\n\n중요: 위의 이전/다음 문맥 정보를 참고하여 자연스럽고 일관된 번역을 제공하세요. 하지만 반드시 주어진 텍스트만 정확하게 번역하세요."
            return f"{base_prompt}{context_section}{context_instruction}"
        
        return base_prompt
    
    async def translate_text_complete(
        self,
        text: str,
        target_lang_cd: str,
        target_lang_name: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        텍스트 번역을 한번에 처리하는 통합 메서드
        1. 텍스트를 청크로 분할
        2. 청크들을 병렬로 번역 (프롬프트와 메시지 기반 LLM 호출)
        3. 번역된 청크들을 합치기
        
        Returns:
            dict: {
                "chunks": List[str],  # 분할된 청크 목록
                "chunk_count": int,   # 청크 개수
                "translated_chunks": List[str],  # 번역된 청크 목록
                "final_translation": str  # 최종 번역 결과
            }
        """
        # 1. 텍스트를 청크로 분할
        chunks = self.split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_count = len(chunks)
        
        # 2. 프롬프트 생성 (한번만 생성하여 재사용)
        prompt = create_translation_prompt(target_lang_cd, target_lang_name)
        
        # 3. 청크들을 병렬로 번역 (문맥 정보 포함)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def translate_with_semaphore(
            chunk: str, 
            index: int,
            previous_chunk: str = None,
            next_chunk: str = None
        ) -> tuple[int, str]:
            """세마포어를 사용한 병렬 번역 - 문맥 정보 포함"""
            async with semaphore:
                is_first = (index == 0)
                is_last = (index == len(chunks) - 1)
                translated = await self._translate_chunk(
                    chunk, 
                    prompt,
                    previous_chunk=previous_chunk,
                    next_chunk=next_chunk,
                    is_first=is_first,
                    is_last=is_last
                )
                return index, translated
        
        # 모든 청크를 병렬로 번역 (문맥 정보 포함)
        tasks = []
        for i, chunk in enumerate(chunks):
            previous_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None
            
            tasks.append(translate_with_semaphore(chunk, i, previous_chunk, next_chunk))
        
        results = await asyncio.gather(*tasks)
        
        # 인덱스 순서대로 정렬
        sorted_results = sorted(results, key=lambda x: x[0])
        translated_chunks = [translated for _, translated in sorted_results]
        
        # 4. 번역된 청크들을 합치기 (오버랩 영역 처리)
        final_translation = self._merge_translated_chunks(translated_chunks, chunk_overlap)
        
        # 결과 반환
        result = {
            "chunks": chunks,
            "chunk_count": chunk_count,
            "translated_chunks": translated_chunks,
            "final_translation": final_translation
        }
        
        return result
    
    def _merge_translated_chunks(self, translated_chunks: List[str], overlap: int = 0) -> str:
        """
        번역된 청크들을 합치기
        
        오버랩이 있는 경우, 자연스러운 연결을 위해 단순 합치기를 수행합니다.
        향후 오버랩 영역의 일관성을 검증하는 로직을 추가할 수 있습니다.
        """
        if not translated_chunks:
            return ""
        
        if len(translated_chunks) == 1:
            return translated_chunks[0]
        
        # 현재는 단순히 공백으로 합치지만, 
        # 오버랩 영역이 있다면 자연스러운 연결을 위해 추가 처리 가능
        # 예: 마지막 문장의 끝 부분과 다음 문장의 시작 부분 검증
        
        # 각 청크 사이에 공백을 추가하여 합치기
        merged = " ".join(translated_chunks)
        
        # 중복된 공백 제거
        merged = re.sub(r'\s+', ' ', merged)
        
        return merged.strip()

