"""번역 오케스트레이터 구현 클래스"""
import time
import os
import json
import asyncio
from asyncio import Queue
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from state.translation_state import TranslationState
from graph.translate_node import TranslateAgent
from utils.language_utils import get_language_name
from orchestrator.base_orchestrator import BaseOrchestrator
from orchestrator.callback_handler import CustomAsyncCallbackHandler
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables import RunnableLambda
from schema.schemas import SSEChunk


class TranslationOrchestrator(BaseOrchestrator):
    """번역 오케스트레이터 구현 클래스 """
    
    def __init__(
        self,
        service_info: Optional[Dict[str, Any]] = None,
        chat_req: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        llm_config: Optional[Dict] = {},
        callbacks: Optional[list] = []
    ):
        """초기화 """
        self.service_info = service_info
        self.chat_req = chat_req
        self.session_id = session_id
        self.queue = Queue()
        self._graph = None
        self.result = None
        self.callbacks = callbacks or []
        
        # LLM 설정
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            streaming=True,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 스트리밍 설정 (환경 변수 읽기를 초기화 시 한 번만 수행)
        self.max_chars_per_event = int(os.getenv("STREAMING_MAX_CHARS_PER_EVENT", "512"))
        self.queue_poll_timeout = float(os.getenv("STREAMING_QUEUE_POLL_TIMEOUT", "0.01"))
        self.queue_max_timeouts = int(os.getenv("STREAMING_QUEUE_MAX_TIMEOUTS", "5"))
    
    def _get_graph(self):
        """그래프 생성 """
        if self._graph:
            return self._graph
        
        TRANSLATE_NODE = "TRANSLATE_NODE"
        sg = StateGraph(TranslationState)
        
        # node 생성
        sg.add_node(
            TRANSLATE_NODE,
            RunnableLambda(
                TranslateAgent(
                    service_info=self.service_info,
                    chat_req=self.chat_req,
                    llm=self.llm,
                    callbacks=self.callbacks
                )
            )
        )
        sg.add_edge(START, TRANSLATE_NODE)
        sg.add_edge(TRANSLATE_NODE, END)
        
        self._graph = sg.compile()
        return self._graph
    
    async def translate_text(
        self,
        text: str,
        target_lang_cd: str,
        stream: bool = False
    ) -> str:
        """텍스트를 번역하는 메인 메서드"""
        target_lang_name = get_language_name(target_lang_cd)
        
        # 상태 초기화
        state = TranslationState(
            original_text=text,
            target_lang_cd=target_lang_cd,
            target_lang_name=target_lang_name,
            chunks=[],
            translated_chunks=[],
            final_translation="",
            chunk_count=0
        )
        
        # 그래프 생성 및 실행
        graph = self._get_graph()
        final_state = await graph.ainvoke(state)
        
        return final_state.get("final_translation", "")
    
    async def run(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        스트리밍 방식으로 텍스트 번역 
        """
        # Queue와 customAsyncCallbackHandler 사용 
        self.queue = Queue()
        
        # config 설정 (
        self.callbacks = [CustomAsyncCallbackHandler(self.queue)]
        config = ensure_config({
            "callbacks": self.callbacks,
            "configurable": {
                "callback_handler_class": CustomAsyncCallbackHandler
            }
        })
        
        # 상태 초기화 (message를 original_text로 사용)
        # target_lang_cd는 node의 preprocess에서 처리하므로 여기서는 기본값만 설정
        state = TranslationState(
            original_text=message,
            target_lang_cd="",  # node의 preprocess에서 설정됨
            target_lang_name="",  # node의 preprocess에서 설정됨
            chunks=[],
            translated_chunks=[],
            final_translation="",
            chunk_count=0
        )
        
        graph = self._get_graph()
        
        # time
        start_time = time.time()
        first_token_sent_time = None  # 첫 토큰 전송 시간
        stream_end = False
        _result = None
        
        # 스트리밍 인덱스 (0부터 시작) - response body의 index 필드에 사용
        streaming_index = 0
        
        # "번역중입니다" 상태 메시지를 제일 먼저 전송
        status_message = "번역중입니다"
        sse_chunk = SSEChunk(
            index=streaming_index,
            step="status",
            rspns_msg=status_message,
            cmptn_yn=False,
            tokn_info={},
            link_info=[],
            src_doc_info=[]
        )
        yield sse_chunk.to_msg()
        streaming_index += 1
        
        # 버퍼링 사용하여 청크 인덱스 순서대로 처리
        stream_buffer = {}  # 각 청크별 스트리밍 버퍼 {chunk_index: "텍스트"}
        last_sent_length = {}  # 각 청크별 마지막 전달 길이 {chunk_index: 길이}
        next_expected_index = 0  # 다음에 처리할 청크 인덱스 (0부터 시작, 청크 번호)
        pending_spaces = {}  # 각 청크별 대기 중인 띄어쓰기 {chunk_index: " "}
        max_chars_per_event = self.max_chars_per_event
        queue_poll_timeout = self.queue_poll_timeout
        queue_max_timeouts = self.queue_max_timeouts
        
        # astream_events로 그래프 실행 
        import logging
        logger = logging.getLogger(__name__)
        event_count = 0
        
        async def drain_queue_to_buffer(allow_wait: bool):
            """Queue에 쌓인 토큰을 하나씩 처리하고 버퍼에 저장"""
            nonlocal stream_buffer, pending_spaces, next_expected_index, last_sent_length
            consecutive_empty_checks = 0
            max_empty_checks = queue_max_timeouts if allow_wait else 1
            
            while consecutive_empty_checks < max_empty_checks:
                # 큐가 비어있는지 먼저 확인
                if self.queue.empty():
                    if allow_wait:
                        consecutive_empty_checks += 1
                        if consecutive_empty_checks >= max_empty_checks:
                            break
                        await asyncio.sleep(queue_poll_timeout)
                        continue
                    else:
                        # allow_wait=False이면 즉시 종료
                        break
                
                # 큐에 데이터가 있으면 하나씩 가져오기
                try:
                    chunk = self.queue.get_nowait()
                except:
                    # 그 사이에 비었으면 다시 체크
                    if allow_wait:
                        consecutive_empty_checks += 1
                        await asyncio.sleep(queue_poll_timeout)
                        continue
                    else:
                        break
                
                # 데이터를 성공적으로 가져왔으면 카운터 리셋
                consecutive_empty_checks = 0
                
                if chunk is None:
                    continue
                
                if isinstance(chunk, tuple):
                    chunk_index, token = chunk
                else:
                    chunk_index, token = (0, chunk)
                
                # None이거나 빈 문자열이면 스킵
                if token is None or token == "":
                    continue
                
                if chunk_index not in stream_buffer:
                    stream_buffer[chunk_index] = ""
                    pending_spaces[chunk_index] = ""
                    last_sent_length[chunk_index] = 0
                
                # 띄어쓰기 처리
                if token == ' ':
                    pending_spaces[chunk_index] = ' '
                else:
                    if pending_spaces.get(chunk_index):
                        stream_buffer[chunk_index] += pending_spaces[chunk_index]
                        pending_spaces[chunk_index] = ""
                    stream_buffer[chunk_index] += token
                
                # 첫 토큰 이후에는 즉시 drain 모드로 전환
                if allow_wait:
                    allow_wait = False
                    max_empty_checks = 1
        
        # graph.astream_events 시작 전에 Queue 확인 (첫 토큰 빠른 처리)
        # 첫 이벤트가 발생하기 전에 Queue에 토큰이 들어올 수 있음
        for _ in range(5):  # 최대 5번 확인 (약 0.05초)
            if not self.queue.empty():
                await drain_queue_to_buffer(allow_wait=False)
                if next_expected_index in stream_buffer and len(stream_buffer[next_expected_index]) > 0:
                    break  # 첫 토큰이 있으면 이벤트 루프로 진입
            await asyncio.sleep(0.01)
        
        async for event in graph.astream_events(state, config=config, version="v2"):
            event_count += 1
            event_name = event.get("event", "unknown")
            
            # LLM 관련 이벤트에서만 큐를 대기하면서 확인
            is_llm_event = (
                event_name.startswith("on_llm_") or
                event_name.startswith("on_chat_")
            )
            
            # 모든 이벤트에서 Queue 확인 (첫 토큰 빠른 처리)
            # Queue가 비어있지 않거나 LLM 이벤트인 경우 drain 실행
            if not self.queue.empty() or is_llm_event:
                await drain_queue_to_buffer(is_llm_event)
            # Queue가 비어있어도 LLM 이벤트가 아니면 한 번 확인 (첫 토큰 빠른 처리)
            elif not is_llm_event:
                await drain_queue_to_buffer(allow_wait=False)
            
            # 버퍼에서 순서대로 전송 (청크 순서 보장, 한 글자씩)
            # drain_queue_to_buffer에서 전송하지 못한 버퍼 내용도 확인
            # next_expected_index와 일치하는 청크만 전송 (순서 보장)
            # 버퍼에 누적된 내용이 있으면 한 글자씩 전송 (여러 글자가 있어도 한 글자씩만)
            while next_expected_index in stream_buffer:
                current_buffer = stream_buffer[next_expected_index]
                last_length = last_sent_length.get(next_expected_index, 0)
                
                # 새로 추가된 내용이 있는지 확인
                if len(current_buffer) > last_length:
                    # 한 글자씩 전송 (자연스러운 스트리밍)
                    chunk_to_send = current_buffer[last_length:last_length+1]
                    if chunk_to_send:
                        # 첫 토큰 전송 시간 측정
                        if first_token_sent_time is None:
                            first_token_sent_time = time.time()
                            elapsed = first_token_sent_time - start_time
                            logger.info(f"⚡ [첫 토큰 전송] {elapsed:.3f}초 소요")
                            print(f"⚡ [첫 토큰 전송] {elapsed:.3f}초 소요")
                        
                        sse_chunk = SSEChunk(
                            index=streaming_index,
                            step="message",
                            rspns_msg=chunk_to_send,  # 한 글자씩 전송
                            cmptn_yn=False,
                            tokn_info={},
                            link_info=[],
                            src_doc_info=[]
                        )
                        yield sse_chunk.to_msg()
                        streaming_index += 1
                        last_sent_length[next_expected_index] = last_length + 1
                        
                        # 한 번에 하나만 전송하고 break (다음 이벤트에서 계속)
                        break
                else:
                    # 현재 버퍼를 모두 전송했으면 다음 인덱스로
                    next_expected_index += 1
            
            
            # 이벤트 처리
            if event["event"] == "on_chat_model_end":
                # 청크 완료 감지 
                # 대신 모든 청크가 완료되었는지 확인하는 방식 사용
                pass
            
            elif event["event"] == "on_chain_end":
                try:
                    _result = event["data"]
                    # 모든 청크가 완료되었는지 확인
                    if _result and isinstance(_result, dict) and "output" in _result:
                        output = _result["output"]
                        if isinstance(output, dict) and "answer" in output:
                            # answer가 있으면 번역이 완료된 것으로 간주
                            # 남은 Queue의 토큰을 버퍼로 옮기기 (타임아웃 에러 방지)
                            consecutive_empty = 0
                            max_empty = 10  # 연속으로 비어있는 경우 최대 10번
                            
                            while consecutive_empty < max_empty:
                                # 큐가 비어있지 않으면 즉시 처리
                                if not self.queue.empty():
                                    consecutive_empty = 0  # 리셋
                                    await drain_queue_to_buffer(allow_wait=False)
                                else:
                                    # 큐가 비어있으면 대기 후 다시 확인
                                    consecutive_empty += 1
                                    await asyncio.sleep(0.01)  # 10ms 대기
                            
                            # 버퍼에서 남은 내용 순서대로 전송 (한 번에 하나의 토큰씩, 순서 보장)
                            # next_expected_index와 일치하는 청크만 전송 (순서 보장)
                            # next_expected_index가 없으면 건너뛰기 (순서가 맞지 않은 청크는 나중에 처리)
                            if next_expected_index in stream_buffer:
                                current_buffer = stream_buffer[next_expected_index]
                                last_length = last_sent_length.get(next_expected_index, 0)
                                
                                if len(current_buffer) > last_length:
                                    # 한 번에 하나의 문자만 전송 (자연스러운 스트리밍)
                                    chunk_to_send = current_buffer[last_length:last_length+1]
                                    if chunk_to_send:
                                        sse_chunk = SSEChunk(
                                            index=streaming_index,
                                            step="message",
                                            rspns_msg=chunk_to_send,  # 한 번에 하나씩 전송
                                            cmptn_yn=False,
                                            tokn_info={},
                                            link_info=[],
                                            src_doc_info=[]
                                        )
                                        yield sse_chunk.to_msg()
                                        streaming_index += 1
                                        last_sent_length[next_expected_index] = last_length + 1
                                else:
                                    # 현재 버퍼를 모두 전송했으면 다음 인덱스로
                                    next_expected_index += 1
                            
                            # _result 저장
                            _result = event["data"]
                except Exception as e:
                    logger.error(f"[on_chain_end] 처리 중 오류: {e}")
                    continue
            
            elif event["event"] == "on_chain_start":
                # 상태 메시지는 이미 run 메서드 시작 시 전송하므로 여기서는 스킵
                # 필요시 다른 이벤트 처리 로직 추가 가능
                continue
        
        # 루프 종료 후 남은 큐 처리 (순서 보장, 타임아웃 에러 방지)
        consecutive_empty = 0
        max_empty = 10  # 연속으로 비어있는 경우 최대 10번
        
        while consecutive_empty < max_empty:
            # 큐가 비어있지 않으면 즉시 처리
            if not self.queue.empty():
                consecutive_empty = 0  # 리셋
                await drain_queue_to_buffer(allow_wait=False)
            else:
                # 큐가 비어있으면 대기 후 다시 확인
                consecutive_empty += 1
                await asyncio.sleep(0.01)  # 10ms 대기
        
        # 버퍼에서 남은 내용 순서대로 전송 (한 번에 하나의 토큰씩, 순서 보장)
        # next_expected_index부터 순서대로만 전송 (순서가 맞지 않은 청크는 건너뛰기)
        max_iterations = 1000  # 무한 루프 방지
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # next_expected_index가 버퍼에 없으면 건너뛰기 (순서가 맞지 않은 청크는 나중에 처리)
            if next_expected_index not in stream_buffer:
                # 다음 인덱스가 있는지 확인 (순서가 맞지 않은 청크가 있을 수 있음)
                # 하지만 순서 보장을 위해 next_expected_index만 전송
                break
            
            current_buffer = stream_buffer[next_expected_index]
            last_length = last_sent_length.get(next_expected_index, 0)
            
            if len(current_buffer) > last_length:
                # 한 번에 하나의 문자만 전송 (자연스러운 스트리밍)
                chunk_to_send = current_buffer[last_length:last_length+1]
                if chunk_to_send:
                    sse_chunk = SSEChunk(
                        index=streaming_index,
                        step="message",
                        rspns_msg=chunk_to_send,  # 한 번에 하나씩 전송
                        cmptn_yn=False,
                        tokn_info={},
                        link_info=[],
                        src_doc_info=[]
                    )
                    yield sse_chunk.to_msg()
                    streaming_index += 1
                    last_sent_length[next_expected_index] = last_length + 1
                    # 한 번에 하나만 전송하고 계속 (루프 종료 후이므로 모두 전송)
                    continue
            else:
                # 현재 버퍼를 모두 전송했으면 다음 인덱스로
                next_expected_index += 1
                continue
        
        # _result 안전하게 처리
        result = None
        if _result and isinstance(_result, dict) and "output" in _result:
            result = _result["output"]
        self.result = result
        
        # final 메시지 전송 (DataResponse 구조에 맞춰)
        if result:
            final_answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            sse_chunk = SSEChunk(
                index=-1,
                step="final",
                rspns_msg=final_answer,  # answer alias
                cmptn_yn=True,  # completion alias
                tokn_info={},  # TokenInfo 객체 (기본값)
                link_info=[],
                src_doc_info=[]
            )
            yield sse_chunk.to_msg()
