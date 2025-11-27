"""번역 오케스트레이터 구현 클래스"""
import json
import os
import asyncio
from asyncio import Queue
from typing import Optional, Dict, Any, Any as AnyType
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from state.translation_state import TranslationState
from graph.translate_node import translate_node
from utils.language_utils import get_language_name
from orchestrator.base_orchestrator import BaseOrchestrator
from langchain_core.runnables import ensure_config
from schema.schemas import SSEChunk


class TranslationOrchestrator(BaseOrchestrator):
    """번역 오케스트레이터 구현 클래스"""
    
    def __init__(self):
        """초기화"""
        self._graph = None
        self._llm = None
    
    def _create_llm(self) -> ChatOpenAI:
        """LLM 인스턴스 생성"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.3,
                streaming=True,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        return self._llm
    
    def _create_graph(self):
        """번역을 위한 LangGraph 생성 및 구성"""
        if self._graph is None:
            workflow = StateGraph(TranslationState)
            
            # translate_node.py의 translate_node 함수를 노드로 직접 연결
            workflow.add_node("translate", translate_node)
            workflow.set_entry_point("translate")
            workflow.add_edge("translate", END)
            self._graph = workflow.compile()
        
        return self._graph
    
    def _create_initial_state(
        self,
        text: str,
        target_lang_cd: str,
        target_lang_name: str
    ) -> TranslationState:
        """초기 상태 생성"""
        return {
            "original_text": text,
            "target_lang_cd": target_lang_cd,
            "target_lang_name": target_lang_name,
            "chunks": [],
            "translated_chunks": [],
            "final_translation": "",
            "chunk_count": 0
        }
    
    async def translate_text(
        self,
        text: str,
        target_lang_cd: str,
        stream: bool = False
    ) -> str:
        """텍스트를 번역하는 메인 메서드"""
        target_lang_name = get_language_name(target_lang_cd)
        
        # 상태 초기화
        initial_state = self._create_initial_state(text, target_lang_cd, target_lang_name)
        
        # 그래프 생성 및 실행
        graph = self._create_graph()
        final_state = await graph.ainvoke(initial_state)
        
        return final_state["final_translation"]
    
    async def translate_text_stream(
        self,
        text: str,
        target_lang_cd: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        스트리밍 방식으로 텍스트 번역 (astream_events + Queue 사용)
        사용자 코드 구조 사용
        """
        from langchain_core.callbacks import AsyncCallbackHandler
        
        # Queue와 customAsyncCallbackHandler 사용 (사용자 코드 구조)
        self.queue = Queue()
        
        # customAsyncCallbackHandler 클래스 정의 (이미지 코드 그대로 - 절대 수정하지 않음)
        class CustomAsyncCallbackHandler(AsyncCallbackHandler):
            def __init__(self, queue: Queue):
                super().__init__()
                self.queue = queue
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self.total_tokens = 0
            
            async def on_llm_new_token(self, token: str, **kwargs) -> None:
                """LLM 토큰 스트리밍 시 호출"""
                # 이미지 코드 그대로
                if len(kwargs["chunk"].message.response_metadata):
                    self.prompt_tokens += kwargs["chunk"].message.response_metadata["usage"]["prompt_tokens"]
                    self.completion_tokens += kwargs["chunk"].message.response_metadata["usage"]["completion_tokens"]
                    self.total_tokens += kwargs["chunk"].message.response_metadata["usage"]["total_tokens"]
                    await self.queue.put(None)
                else:
                    await self.queue.put(token)
            
            async def on_llm_end(self, response, **kwargs: Any) -> None:
                """LLM 응답 종료 시 호출"""
                # 이미지 코드 그대로
                self.prompt_tokens += response.generations[0][0].message.response_metadata["usage"]["prompt_tokens"]
                self.completion_tokens += response.generations[0][0].message.response_metadata["usage"]["completion_tokens"]
                self.total_tokens += response.generations[0][0].message.response_metadata["usage"]["total_tokens"]
        
        # config에 Queue와 CustomAsyncCallbackHandler 클래스 전달 (translate_node에서 사용)
        # translate_node에서 각 청크마다 인덱스별 callback을 생성하므로, 클래스와 Queue를 전달
        self.callbacks = [CustomAsyncCallbackHandler(self.queue)]  # Queue 접근용 기본 callback
        run_config = ensure_config({
            "callbacks": self.callbacks,
            "configurable": {
                "streaming_queue": self.queue,
                "callback_handler_class": CustomAsyncCallbackHandler
            }
        })
        
        target_lang_name = get_language_name(target_lang_cd)
        initial_state = self._create_initial_state(text, target_lang_cd, target_lang_name)
        graph = self._create_graph()
        
        # 이전 코드 구조 (버퍼링 사용하여 순서 보장)
        stream_buffer = {}  # 각 청크별 스트리밍 버퍼
        last_sent_length = {}  # 각 청크별 마지막 전달 길이
        next_expected_index = 1  # 다음에 전달할 인덱스 (1부터 시작)
        streaming_index = 1  # 스트리밍 인덱스 (1부터 시작, final일 때 -1)
        stream_end = False
        _result = None
        
        # astream_events로 그래프 실행 (이전 코드 구조)
        async for event in graph.astream_events(initial_state, version="v2", config=run_config):
            event_type = event.get("event")
            
            if event_type == "on_chat_model_stream":
                # Queue에서 chunk 가져오기 (이전 코드 구조)
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                    if not item:
                        stream_end = True
                        continue
                    
                    # (chunk_index, token) 튜플 형태
                    chunk_index, token = item if isinstance(item, tuple) else (1, item)
                    if token is None:
                        self.queue.task_done()
                        continue
                    
                    # 버퍼에 저장
                    if chunk_index not in stream_buffer:
                        stream_buffer[chunk_index] = ""
                    stream_buffer[chunk_index] += token
                    
                    # 인덱스 순서대로 전달 (1번부터)
                    while next_expected_index in stream_buffer:
                        current_buffer = stream_buffer[next_expected_index]
                        last_length = last_sent_length.get(next_expected_index, 0)
                        
                        if len(current_buffer) > last_length:
                            new_content = current_buffer[last_length:]
                            # SSEChunk 사용 (step: "delta"는 to_msg()에서 "message"로 변환됨)
                            # answer에 JSON 문자열로 필요한 정보 포함
                            answer_data = {
                                "content": new_content,
                                "streaming_index": streaming_index,
                                "chunk_index": next_expected_index
                            }
                            sse_chunk = SSEChunk(
                                step="delta",  # "delta"는 to_msg()에서 event: "message"로 변환, step 필드는 "delta"로 유지
                                answer=json.dumps(answer_data, ensure_ascii=False)
                            )
                            yield sse_chunk.to_msg()
                            last_sent_length[next_expected_index] = len(current_buffer)
                            streaming_index += 1
                        
                        next_expected_index += 1
                    
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    continue
                continue
            
            elif event_type == "on_chat_model_end":
                stream_end = True
                continue
            
            elif event_type == "on_chain_end":
                _result = event.get("data", {})
                if stream_end:
                    break
            
            elif event_type == "on_chain_start":
                metadata = event.get("metadata", {})
                tags = event.get("tags", [])
                
                if "langgraph_node" in metadata.keys():
                    if len(tags) > 0 and tags[0] != "seq:step:1":
                        continue
                    if metadata["langgraph_node"] == "TRANSLATE_NODE":
                        status_message = "번역중.."
                        sse_chunk = SSEChunk(step="status", answer=status_message, completion=False)
                        yield sse_chunk.to_msg()
                    else:
                        continue
        
        # 남은 버퍼 처리
        while next_expected_index in stream_buffer:
            current_buffer = stream_buffer[next_expected_index]
            last_length = last_sent_length.get(next_expected_index, 0)
            
            if len(current_buffer) > last_length:
                new_content = current_buffer[last_length:]
                answer_data = {
                    "content": new_content,
                    "streaming_index": streaming_index,
                    "chunk_index": next_expected_index
                }
                sse_chunk = SSEChunk(
                    step="delta",
                    answer=json.dumps(answer_data, ensure_ascii=False)
                )
                yield sse_chunk.to_msg()
                streaming_index += 1
            
            next_expected_index += 1
        
        # 결과 추출 및 final 전송
        if _result:
            result = _result.get("output", {})
            self.result = result
            final_answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            final_data = {
                "content": final_answer,
                "streaming_index": -1,
                "chunk_index": -1
            }
            final_chunk = SSEChunk(
                step="final",
                answer=json.dumps(final_data, ensure_ascii=False),
                completion=True
            )
            yield final_chunk.to_msg()
