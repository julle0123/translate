"""번역 오케스트레이터 구현 클래스"""
import time
import os
import json
import asyncio
from asyncio import Queue
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.graph import CompiledStateGraph

from state.translation_state import TranslationState
from graph.translate_node import TranslateAgent
from utils.language_utils import get_language_name
from orchestrator.base_orchestrator import BaseOrchestrator
from orchestrator.callback_handler import CustomAsyncCallbackHandler
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables import RunnableLambda
from schema.schemas import SSEChunk


class TranslationOrchestrator(BaseOrchestrator):
    """번역 오케스트레이터 구현 클래스 (이미지 코드 구조)"""
    
    def __init__(
        self,
        service_info: Optional[Dict[str, Any]] = None,
        chat_req: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        llm_config: Optional[Dict] = {},
        callbacks: Optional[list] = []
    ):
        """초기화 (이미지 코드 구조)"""
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
    
    def _get_graph(self) -> CompiledStateGraph:
        """그래프 생성 (이미지 코드 구조)"""
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
        스트리밍 방식으로 텍스트 번역 (이미지 코드 구조)
        """
        # Queue와 customAsyncCallbackHandler 사용 (이미지 코드 구조)
        self.queue = Queue()
        
        # config 설정 (이미지 코드 구조)
        self.callbacks = [CustomAsyncCallbackHandler(self.queue)]
        config = ensure_config({
            "callbacks": self.callbacks,
            "configurable": {
                "callback_handler_class": CustomAsyncCallbackHandler
            }
        })
        
        # 상태 초기화 (message를 original_text로 사용)
        # target_lang_cd는 service_info에서 가져오거나 기본값 사용
        target_lang_cd = self.service_info.get("lang_cd", "en") if self.service_info else "en"
        target_lang_name = get_language_name(target_lang_cd)
        state = TranslationState(
            original_text=message,
            target_lang_cd=target_lang_cd,
            target_lang_name=target_lang_name,
            chunks=[],
            translated_chunks=[],
            final_translation="",
            chunk_count=0
        )
        
        graph = self._get_graph()
        
        # time
        start_time = time.time()
        stream_end = False
        _result = None
        
        # 스트리밍 인덱스 (0부터 시작) - response body의 index 필드에 사용
        streaming_index = 0
        
        # 버퍼링 사용하여 청크 인덱스 순서대로 처리
        stream_buffer = {}  # 각 청크별 스트리밍 버퍼 {chunk_index: "텍스트"}
        last_sent_length = {}  # 각 청크별 마지막 전달 길이 {chunk_index: 길이}
        next_expected_index = 0  # 다음에 처리할 청크 인덱스 (0부터 시작, 청크 번호)
        pending_spaces = {}  # 각 청크별 대기 중인 띄어쓰기 {chunk_index: " "}
        
        # Queue 폴링을 위한 플래그
        queue_poller_running = True
        graph_completed = False
        
        # Queue 폴링 태스크: Queue에서 토큰을 받아서 버퍼에 저장
        async def queue_poller():
            """Queue에서 토큰을 지속적으로 폴링하여 버퍼에 저장"""
            nonlocal queue_poller_running
            while queue_poller_running:
                try:
                    # 타임아웃을 두어 주기적으로 체크
                    chunk = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    if chunk is None:
                        continue
                    
                    # (chunk_index, token) 튜플 형태
                    if isinstance(chunk, tuple):
                        chunk_index, token = chunk
                    else:
                        chunk_index, token = (0, chunk)
                    
                    if token is None:
                        continue
                    
                    # 버퍼 초기화
                    if chunk_index not in stream_buffer:
                        stream_buffer[chunk_index] = ""
                        pending_spaces[chunk_index] = ""
                    
                    # 띄어쓰기 처리: 띄어쓰기만 오면 대기, 다음 토큰과 함께 처리
                    if token == ' ':
                        # 띄어쓰기만 있으면 대기
                        pending_spaces[chunk_index] = ' '
                    else:
                        # 일반 토큰인 경우, 대기 중인 띄어쓰기가 있으면 함께 추가
                        if pending_spaces.get(chunk_index):
                            stream_buffer[chunk_index] += pending_spaces[chunk_index]
                            pending_spaces[chunk_index] = ""
                        stream_buffer[chunk_index] += token
                except asyncio.TimeoutError:
                    # 타임아웃은 정상 (Queue가 비어있을 때)
                    continue
                except Exception as e:
                    # 에러 발생 시 로깅하고 계속
                    import logging
                    logging.error(f"Queue poller error: {e}")
                    continue
        
        # Queue 폴링 태스크 시작
        poller_task = asyncio.create_task(queue_poller())
        
        # 청크 완료 추적
        chunk_completion_status = {}  # {chunk_index: bool}
        expected_chunk_count = 0  # 예상 청크 개수 (나중에 업데이트)
        
        # astream_events로 그래프 실행 (이미지 코드 구조)
        async for event in graph.astream_events(state, config=config):
            # 버퍼에서 순서대로 처리하는 로직
            # next_expected_index부터 순서대로 처리
            while next_expected_index in stream_buffer:
                current_buffer = stream_buffer[next_expected_index]
                last_length = last_sent_length.get(next_expected_index, 0)
                
                # 전달할 내용이 있는지 확인
                if len(current_buffer) > last_length:
                    # 현재 위치부터 읽기 시작
                    remaining = current_buffer[last_length:]
                    
                    # 띄어쓰기로 시작하는 경우 다음 글자까지 읽기
                    if remaining.startswith(' '):
                        # 다음 글자가 있으면 함께 보내기
                        if len(remaining) > 1:
                            # 띄어쓰기 + 다음 글자
                            chunk_to_send = remaining[:2]
                            last_sent_length[next_expected_index] = last_length + 2
                        else:
                            # 다음 글자가 아직 없으면 대기 (다음 이벤트에서 처리)
                            break
                    else:
                        # 일반 글자는 한 글자씩
                        chunk_to_send = remaining[0]
                        last_sent_length[next_expected_index] = last_length + 1
                    
                    # DataResponse 구조에 맞춰 SSEChunk 생성 (answer는 문자열만)
                    sse_chunk = SSEChunk(
                        index=streaming_index,
                        step="message",
                        rspns_msg=chunk_to_send,  # answer alias - 문자열만
                        cmptn_yn=False,  # completion alias
                        tokn_info={},  # TokenInfo 객체 (기본값)
                        link_info=[],
                        src_doc_info=[]
                    )
                    yield sse_chunk.to_msg()
                    streaming_index += 1
                    
                    # 한 글자 전달하고 break (다음 이벤트에서 계속)
                    break
                else:
                    # 현재 인덱스의 버퍼가 모두 전달됨 → 다음 인덱스로
                    next_expected_index += 1
            
            # 이벤트 처리
            if event["event"] == "on_chat_model_end":
                # 청크 완료 감지 (metadata에서 청크 인덱스 추출 시도)
                # 주의: LangChain 이벤트에서 청크 인덱스를 직접 추출하기 어려울 수 있음
                # 대신 모든 청크가 완료되었는지 확인하는 방식 사용
                pass
            
            elif event["event"] == "on_chain_end":
                _result = event["data"]
                graph_completed = True
                # 모든 청크가 완료되었는지 확인
                # translate_node에서 state['answer']에 결과가 들어가면 완료로 간주
                if _result and isinstance(_result, dict) and "output" in _result:
                    output = _result["output"]
                    if isinstance(output, dict) and "answer" in output:
                        # answer가 있으면 번역이 완료된 것으로 간주
                        break
            
            elif event["event"] == "on_chain_start":
                if "langgraph_node" in event.get("metadata", {}).keys():
                    if event.get("tags", [""])[0] != "seq:step:1":
                        continue
                    if event["metadata"]["langgraph_node"] == "TRANSLATE_NODE":
                        status_message = "번역중.."
                        # DataResponse 구조에 맞춰 SSEChunk 생성
                        yield SSEChunk(
                            index=-1,
                            step="status",
                            rspns_msg=status_message,  # answer alias
                            cmptn_yn=False,  # completion alias
                            tokn_info={},  # TokenInfo 객체 (기본값)
                            link_info=[],
                            src_doc_info=[]
                        )
                    else:
                        continue
        
        # Queue 폴링 태스크 종료
        queue_poller_running = False
        poller_task.cancel()
        try:
            await poller_task
        except asyncio.CancelledError:
            pass
        
        # 남은 버퍼 처리 (띄어쓰기는 다음 단어와 묶어서)
        # streaming_index는 계속 증가 (이미 설정된 값 유지)
        while next_expected_index in stream_buffer:
            current_buffer = stream_buffer[next_expected_index]
            last_length = last_sent_length.get(next_expected_index, 0)
            
            # 전달할 내용이 있는지 확인
            while len(current_buffer) > last_length:
                current_char = current_buffer[last_length]
                chunk_to_send = None
                
                # 띄어쓰기인 경우 다음 글자까지 읽어서 함께 보내기
                if current_char == ' ':
                    # 다음 글자가 있으면 함께 보내기
                    if len(current_buffer) > last_length + 1:
                        chunk_to_send = current_buffer[last_length:last_length + 2]
                        last_length += 2
                    else:
                        # 다음 글자가 없으면 띄어쓰기만 보내기
                        chunk_to_send = current_char
                        last_length += 1
                else:
                    # 일반 글자는 한 글자씩
                    chunk_to_send = current_char
                    last_length += 1
                
                # chunk_to_send가 정의된 경우에만 전달
                if chunk_to_send:
                    # DataResponse 구조에 맞춰 SSEChunk 생성 (answer는 문자열만)
                    sse_chunk = SSEChunk(
                        index=streaming_index,
                        step="message",
                        rspns_msg=chunk_to_send,  # answer alias - 문자열만
                        cmptn_yn=False,  # completion alias
                        tokn_info={},  # TokenInfo 객체 (기본값)
                        link_info=[],
                        src_doc_info=[]
                    )
                    yield sse_chunk.to_msg()
                    streaming_index += 1
            
            # last_sent_length 업데이트
            # last_length는 while 루프 시작 전에 초기화되고, 루프 안에서 계속 업데이트됨
            last_sent_length[next_expected_index] = last_length
            next_expected_index += 1
        
        result = _result["output"]
        self.result = result
        
        # final 메시지 전송 (DataResponse 구조에 맞춰)
        if result:
            final_answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            yield SSEChunk(
                index=-1,
                step="final",
                rspns_msg=final_answer,  # answer alias
                cmptn_yn=True,  # completion alias
                tokn_info={},  # TokenInfo 객체 (기본값)
                link_info=[],
                src_doc_info=[]
            )
