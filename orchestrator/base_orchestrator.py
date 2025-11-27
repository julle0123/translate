"""Orchestrator 추상 기본 클래스"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator


class BaseOrchestrator(ABC):
    """번역 오케스트레이터의 추상 기본 클래스"""
    
    @abstractmethod
    async def translate_text(
        self,
        text: str,
        target_lang_cd: str,
        stream: bool = False
    ) -> str:
        """
        텍스트를 번역하는 추상 메서드
        
        Args:
            text: 번역할 텍스트
            target_lang_cd: 대상 언어 코드
            stream: 스트리밍 사용 여부
        
        Returns:
            str: 번역된 텍스트
        """
        pass
    
    @abstractmethod
    async def translate_text_stream(
        self,
        text: str,
        target_lang_cd: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 방식으로 텍스트 번역하는 추상 메서드
        
        Args:
            text: 번역할 텍스트
            target_lang_cd: 대상 언어 코드
            config: 추가 설정 (선택사항)
        
        Yields:
            str: 스트리밍되는 번역 결과
        """
        pass

