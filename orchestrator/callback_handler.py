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
        LLM 토큰 스트리밍 시 호출 
        
        Args:
            token: 스트리밍되는 토큰
            **kwargs: 추가 인자
        """
        # 기존 코드 (주석 처리)
        # if len(kwargs["chunk"].message.response_metadata):
        #     self.prompt_tokens += kwargs["chunk"].message.response_metadata["usage"]["prompt_tokens"]
        #     self.completion_tokens += kwargs["chunk"].message.response_metadata["usage"]["completion_tokens"]
        #     self.total_tokens += kwargs["chunk"].message.response_metadata["usage"]["total_tokens"]
        #     await self.queue.put(None)
        # else:
        #     await self.queue.put(token)
        
        # 새로운 코드 (usage 안전 처리)
        if kwargs.get("chunk") and kwargs["chunk"].message.response_metadata:
            usage = kwargs["chunk"].message.response_metadata.get("usage", {})
            if usage:
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)
                await self.queue.put(None)
            else:
                await self.queue.put(token)
        else:
            await self.queue.put(token)
    
    async def on_llm_end(self, response, **kwargs: Any) -> None:
        """
        LLM 응답 종료 시 호출 
        
        Args:
            response: LLM 응답
            **kwargs: 추가 인자
        """
        # 기존 코드 (주석 처리)
        # self.prompt_tokens += response.generations[0][0].message.response_metadata["usage"]["prompt_tokens"]
        # self.completion_tokens += response.generations[0][0].message.response_metadata["usage"]["completion_tokens"]
        # self.total_tokens += response.generations[0][0].message.response_metadata["usage"]["total_tokens"]
        
        # 새로운 코드 (usage 안전 처리)
        try:
            if (response.generations and 
                len(response.generations) > 0 and 
                len(response.generations[0]) > 0 and
                response.generations[0][0].message.response_metadata):
                usage = response.generations[0][0].message.response_metadata.get("usage", {})
                if usage:
                    self.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.completion_tokens += usage.get("completion_tokens", 0)
                    self.total_tokens += usage.get("total_tokens", 0)
        except (KeyError, IndexError, AttributeError):
            # usage가 없거나 구조가 다를 경우 무시
            pass

