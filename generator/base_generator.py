"""Generator 추상 기본 클래스"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_openai import ChatOpenAI


class BaseGenerator(ABC):
    """번역 생성기의 추상 기본 클래스"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        Args:
            llm: 사용할 LLM 모델
        """
        self.llm = llm
    
    @abstractmethod
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
        텍스트 번역을 한번에 처리하는 추상 메서드
        
        Args:
            text: 번역할 원본 텍스트
            target_lang_cd: 대상 언어 코드
            target_lang_name: 대상 언어 이름
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            max_concurrent: 최대 동시 처리 수
        
        Returns:
            dict: {
                "chunks": List[str],  # 분할된 청크 목록
                "chunk_count": int,   # 청크 개수
                "translated_chunks": List[str],  # 번역된 청크 목록
                "final_translation": str  # 최종 번역 결과
            }
        """
        pass

