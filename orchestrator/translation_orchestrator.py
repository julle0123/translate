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
            """Queue에 쌓인 토큰을 버퍼로 옮기고 필요 시 잠시 대기"""
            nonlocal stream_buffer, pending_spaces
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
                
                # 큐에 데이터가 있으면 가져오기
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
                
                if token is None:
                    continue
                
                if chunk_index not in stream_buffer:
                    stream_buffer[chunk_index] = ""
                    pending_spaces[chunk_index] = ""
                
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
        
        async for event in graph.astream_events(state, config=config, version="v2"):
            event_count += 1
            event_name = event.get("event", "unknown")
            
            # LLM 관련 이벤트에서만 큐를 대기하면서 확인
            is_llm_event = (
                event_name.startswith("on_llm_") or
                event_name.startswith("on_chat_")
            )
            
            # 큐가 비어있지 않거나 대기가 필요한 경우에만 drain 실행
            if not self.queue.empty() or is_llm_event:
                await drain_queue_to_buffer(is_llm_event)
            
            # 모든 이벤트에서 버퍼 처리 로직 실행 
            # next_expected_index부터 순서대로 처리
            buffer_processed = False
            processed_chars_this_event = 0
            
            # LLM 이벤트가 아니면 배치 제한 없이 버퍼를 최대한 비움
            effective_max_chars = max_chars_per_event if is_llm_event else float('inf')
            
            # 버퍼에 데이터가 있으면 계속 처리 (이벤트 타입 무관)
            while next_expected_index in stream_buffer:
                current_buffer = stream_buffer[next_expected_index]
                last_length = last_sent_length.get(next_expected_index, 0)
                
                # 전달할 내용이 있는지 확인
                if len(current_buffer) > last_length:
                    wait_for_more_tokens = False
                    
                    # 청크 0은 즉시 전송 (대기 없음)
                    is_chunk_0 = (next_expected_index == 0)
                    
                    while len(current_buffer) > last_length:
                        # 현재 위치부터 읽기 시작
                        remaining = current_buffer[last_length:]
                        
                        # 띄어쓰기로 시작하는 경우 다음 글자까지 읽기
                        if remaining.startswith(' '):
                            # 다음 글자가 있으면 함께 보내기
                            if len(remaining) > 1:
                                chunk_to_send = remaining[:2]
                                last_length += 2
                            else:
                                # 청크 0이 아니면 다음 글자 대기, 청크 0은 즉시 전송
                                if not is_chunk_0:
                                    wait_for_more_tokens = True
                                    break
                                else:
                                    # 청크 0은 공백만이라도 즉시 전송
                                    chunk_to_send = remaining[0]
                                    last_length += 1
                        else:
                            # 일반 글자는 한 글자씩
                            chunk_to_send = remaining[0]
                            last_length += 1
                        
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
                        buffer_processed = True
                        
                        # 첫 토큰 전송 시간 측정
                        if first_token_sent_time is None and streaming_index == 1:
                            first_token_sent_time = time.time()
                            elapsed = first_token_sent_time - start_time
                            logger.info(f"⚡ [첫 토큰 전송] {elapsed:.3f}초 소요")
                            print(f"⚡ [첫 토큰 전송] {elapsed:.3f}초 소요")
                        last_sent_length[next_expected_index] = last_length
                        processed_chars_this_event += len(chunk_to_send)
                        
                        # 배치 크기 제한 (LLM 이벤트에만 적용)
                        if processed_chars_this_event >= effective_max_chars:
                            break
                    
                    # 다음 토큰을 기다려야 하거나 배치 한도 도달 시 루프 종료
                    if wait_for_more_tokens or processed_chars_this_event >= effective_max_chars:
                        break
                    
                    # 현재 인덱스의 버퍼가 모두 전달됨 → 다음 인덱스로
                    if len(current_buffer) <= last_length:
                        last_sent_length[next_expected_index] = last_length
                        next_expected_index += 1
                    else:
                        break
                else:
                    # 현재 인덱스의 버퍼가 모두 전달됨 → 다음 인덱스로
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
                    # translate_node에서 state['answer']에 결과가 들어가면 완료로 간주
                    if _result and isinstance(_result, dict) and "output" in _result:
                        output = _result["output"]
                        if isinstance(output, dict) and "answer" in output:
                            # answer가 있으면 번역이 완료된 것으로 간주
                            # 남은 Queue의 토큰 처리 (타임아웃을 두어 안전하게 처리)
                            # Queue가 비어있을 수 있으므로 타임아웃 사용
                            import logging
                            logger = logging.getLogger(__name__)
                            
                            remaining_tokens_count = 0
                            consecutive_timeouts = 0
                            max_timeouts = 10  # 연속 타임아웃 최대 횟수
                            
                            # Queue가 완전히 비어질 때까지 처리 (타임아웃이 연속으로 발생할 때까지)
                            while consecutive_timeouts < max_timeouts:
                                try:
                                    # asyncio.wait_for 대신 직접 타임아웃 구현 (예외 체인 문제 방지)
                                    # asyncio.wait_for는 타임아웃 시 CancelledError를 발생시키고 TimeoutError로 래핑함
                                    # 따라서 asyncio.wait를 사용하여 직접 타임아웃 구현
                                    chunk = None
                                    try:
                                        # asyncio.wait를 사용하여 직접 타임아웃 구현
                                        timeout_occurred = False
                                        task = asyncio.create_task(self.queue.get())
                                        done, pending = await asyncio.wait(
                                            [task],
                                            timeout=0.1,
                                            return_when=asyncio.FIRST_COMPLETED
                                        )
                                        
                                        if done:
                                            # 작업이 완료되었으면 결과 가져오기
                                            completed_task = done.pop()
                                            # task가 취소되었는지 확인
                                            if completed_task.cancelled():
                                                # 취소된 경우 타임아웃으로 처리 (예외 발생 없이)
                                                chunk = None
                                                timeout_occurred = True
                                            else:
                                                try:
                                                    chunk = await completed_task
                                                    timeout_occurred = False
                                                except asyncio.CancelledError:
                                                    # task가 취소된 경우 타임아웃으로 처리 (예외 발생 없이)
                                                    chunk = None
                                                    timeout_occurred = True
                                            # pending 작업 취소 (이미 done이므로 pending은 비어있을 것)
                                            for p_task in pending:
                                                p_task.cancel()
                                                # 취소된 task는 await하지 않음
                                        else:
                                            # 타임아웃 발생 - task 취소
                                            task.cancel()
                                            # 취소된 task는 await하지 않음 (CancelledError 방지)
                                            # 타임아웃으로 처리 (예외 발생 없이)
                                            chunk = None
                                            timeout_occurred = True
                                        
                                        # 타임아웃 발생 시 연속 타임아웃 카운터 증가
                                        if timeout_occurred:
                                            consecutive_timeouts += 1
                                            if consecutive_timeouts >= max_timeouts:
                                                # 연속 타임아웃이 최대 횟수에 도달하면 종료
                                                break
                                            continue
                                    except asyncio.CancelledError:
                                        # 예외 발생 시 타임아웃으로 처리 (예외 발생 없이)
                                        chunk = None
                                        consecutive_timeouts += 1
                                        if consecutive_timeouts >= max_timeouts:
                                            break
                                        logger.warning("[on_chain_end] Queue 처리 중 작업 취소됨, 계속 진행")
                                        print("[on_chain_end] Queue 처리 중 작업 취소됨, 계속 진행")
                                        continue
                                    
                                    # chunk가 None이면 continue
                                    if chunk is None:
                                        continue
                                    
                                    consecutive_timeouts = 0  # 타임아웃 카운터 리셋
                                    remaining_tokens_count += 1
                                    
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
                                    
                                    # 띄어쓰기 처리
                                    if token == ' ':
                                        pending_spaces[chunk_index] = ' '
                                    else:
                                        if pending_spaces.get(chunk_index):
                                            stream_buffer[chunk_index] += pending_spaces[chunk_index]
                                            pending_spaces[chunk_index] = ""
                                        stream_buffer[chunk_index] += token
                                except Exception as e:
                                    # 기타 예외 발생 시 로깅하고 종료
                                    logger.error(f"[on_chain_end] Queue 처리 중 예외 발생: {type(e).__name__}: {e}")
                                    print(f"[on_chain_end] Queue 처리 중 예외 발생: {type(e).__name__}: {e}")
                                    break
                            
                            # 디버깅: 버퍼 상태 로깅
                            logger.info(f"[on_chain_end] Queue에서 처리한 남은 토큰 수: {remaining_tokens_count}")
                        logger.info(f"[on_chain_end] 현재 streaming_index: {streaming_index}, next_expected_index: {next_expected_index}")
                        logger.info(f"[on_chain_end] 버퍼 상태: {[(idx, len(buf), last_sent_length.get(idx, 0)) for idx, buf in stream_buffer.items()]}")
                        print(f"[on_chain_end] Queue에서 처리한 남은 토큰 수: {remaining_tokens_count}")
                        print(f"[on_chain_end] 현재 streaming_index: {streaming_index}, next_expected_index: {next_expected_index}")
                        print(f"[on_chain_end] 버퍼 상태: {[(idx, len(buf), last_sent_length.get(idx, 0)) for idx, buf in stream_buffer.items()]}")
                        
                        # Queue에서 남은 토큰을 모두 처리한 후, 버퍼를 즉시 처리
                        # 버퍼에서 순서대로 처리하는 로직
                        processed_count = 0
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
                                    processed_count += 1
                            
                            # last_sent_length 업데이트
                            last_sent_length[next_expected_index] = last_length
                            logger.info(f"[on_chain_end] 청크 {next_expected_index} 처리 완료: 버퍼 길이 {len(current_buffer)}, 전달 길이 {last_length}, 처리된 토큰 수 {processed_count}")
                            print(f"[on_chain_end] 청크 {next_expected_index} 처리 완료: 버퍼 길이 {len(current_buffer)}, 전달 길이 {last_length}, 처리된 토큰 수 {processed_count}")
                            next_expected_index += 1
                        
                        logger.info(f"[on_chain_end] 버퍼 처리 완료: 총 처리된 토큰 수 {processed_count}, 최종 streaming_index: {streaming_index}, 최종 next_expected_index: {next_expected_index}")
                        print(f"[on_chain_end] 버퍼 처리 완료: 총 처리된 토큰 수 {processed_count}, 최종 streaming_index: {streaming_index}, 최종 next_expected_index: {next_expected_index}")
                        break
                except Exception as e:
                    # on_chain_end 처리 중 예외 발생 시 로깅하고 계속 진행
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"[on_chain_end] 이벤트 처리 중 예외 발생: {type(e).__name__}: {e}")
                    print(f"[on_chain_end] 이벤트 처리 중 예외 발생: {type(e).__name__}: {e}")
                    # 예외가 발생해도 외부 루프는 계속 진행
                    continue
            
            elif event["event"] == "on_chain_start":
                if "langgraph_node" in event.get("metadata", {}).keys():
                    if event.get("tags", [""])[0] != "seq:step:1":
                        continue
                    if event["metadata"]["langgraph_node"] == "TRANSLATE_NODE":
                        status_message = "번역중.."
                        # DataResponse 구조에 맞춰 SSEChunk 생성
                        sse_chunk = SSEChunk(
                            index=-1,
                            step="status",
                            rspns_msg=status_message,  # answer alias
                            cmptn_yn=False,  # completion alias
                            tokn_info={},  # TokenInfo 객체 (기본값)
                            link_info=[],
                            src_doc_info=[]
                        )
                        yield sse_chunk.to_msg()
                    else:
                        continue
        
        # 남은 버퍼 처리 (띄어쓰기는 다음 단어와 묶어서)
        # streaming_index는 계속 증가 (이미 설정된 값 유지)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[루프 종료 후] 남은 버퍼 처리 시작: streaming_index={streaming_index}, next_expected_index={next_expected_index}")
        logger.info(f"[루프 종료 후] 버퍼 상태: {[(idx, len(buf), last_sent_length.get(idx, 0)) for idx, buf in stream_buffer.items()]}")
        
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
            logger.info(f"[루프 종료 후] 청크 {next_expected_index} 처리 완료: 버퍼 길이 {len(current_buffer)}, 전달 길이 {last_length}")
            print(f"[루프 종료 후] 청크 {next_expected_index} 처리 완료: 버퍼 길이 {len(current_buffer)}, 전달 길이 {last_length}")
            next_expected_index += 1
        
        logger.info(f"[루프 종료 후] 최종 처리 완료: streaming_index={streaming_index}, next_expected_index={next_expected_index}")
        print(f"[루프 종료 후] 최종 처리 완료: streaming_index={streaming_index}, next_expected_index={next_expected_index}")
        
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
