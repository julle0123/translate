"""CustomAsyncCallbackHandler 클래스 정의"""
from asyncio import Queue
from langchain_core.callbacks import AsyncCallbackHandler
from typing import Any


class CustomAsyncCallbackHandler(AsyncCallbackHandler):
    """커스텀 비동기 콜백 핸들러 (이미지 코드 구조 - 절대 수정하지 않음)"""
    
    def __init__(self, queue: Queue):
        """
        초기화
        
        Args:
            queue: 토큰을 전달할 Queue
        """
        super().__init__()
        self.queue = queue
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        LLM 토큰 스트리밍 시 호출 (이미지 코드 그대로)
        
        Args:
            token: 스트리밍되는 토큰
            **kwargs: 추가 인자
        """
        # 이미지 코드 그대로
        if len(kwargs["chunk"].message.response_metadata):
            self.prompt_tokens += kwargs["chunk"].message.response_metadata["usage"]["prompt_tokens"]
            self.completion_tokens += kwargs["chunk"].message.response_metadata["usage"]["completion_tokens"]
            self.total_tokens += kwargs["chunk"].message.response_metadata["usage"]["total_tokens"]
            await self.queue.put(None)
        else:
            await self.queue.put(token)
    
    async def on_llm_end(self, response, **kwargs: Any) -> None:
        """
        LLM 응답 종료 시 호출 (이미지 코드 그대로)
        
        Args:
            response: LLM 응답
            **kwargs: 추가 인자
        """
        # 이미지 코드 그대로
        self.prompt_tokens += response.generations[0][0].message.response_metadata["usage"]["prompt_tokens"]
        self.completion_tokens += response.generations[0][0].message.response_metadata["usage"]["completion_tokens"]
        self.total_tokens += response.generations[0][0].message.response_metadata["usage"]["total_tokens"]

