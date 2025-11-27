"""번역 오케스트레이터 구현 클래스"""
import json
import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from state.translation_state import TranslationState
from graph.translate_node import translate_node
from utils.language_utils import get_language_name
from orchestrator.base_orchestrator import BaseOrchestrator
from langchain_core.runnables import ensure_config


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
        스트리밍 방식으로 텍스트 번역 (astream_events 사용)
        
        Args:
            text: 번역할 텍스트
            target_lang_cd: 대상 언어 코드
            config: 추가 설정 (선택사항)
        
        Yields:
            str: 스트리밍되는 번역 결과 (SSE 형식)
        """
        target_lang_name = get_language_name(target_lang_cd)
        initial_state = self._create_initial_state(text, target_lang_cd, target_lang_name)
        graph = self._create_graph()
        runnable_config = ensure_config(config or {})
        
        async for event in graph.astream_events(initial_state, version="v2", config=runnable_config):
            event_type = event.get("event")
            name = event.get("name", "")
            
            # 노드 시작
            if event_type == "on_chain_start" and name == "translate":
                yield f"data: {json.dumps({'status': 'started', 'message': '번역 시작'}, ensure_ascii=False)}\n\n"
            
            # 노드 스트리밍 중 (상태 업데이트)
            elif event_type == "on_chain_stream":
                chunk_data = event.get("data", {})
                if "chunk_count" in chunk_data and "translated_chunks" in chunk_data:
                    chunk_count = chunk_data["chunk_count"]
                    translated_count = len(chunk_data["translated_chunks"])
                    yield f"data: {json.dumps({'status': 'progress', 'translated': translated_count, 'total': chunk_count}, ensure_ascii=False)}\n\n"
            
            # LLM 스트리밍 토큰 (실시간 번역)
            elif event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", {})
                content = chunk.content if hasattr(chunk, "content") else chunk.get("content") if isinstance(chunk, dict) else None
                if content:
                    yield f"data: {json.dumps({'status': 'streaming', 'token': content}, ensure_ascii=False)}\n\n"
            
            # 노드 완료
            elif event_type == "on_chain_end" and name == "translate":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict) and output.get("final_translation"):
                    yield f"data: {json.dumps({'status': 'completed', 'translation': output['final_translation']}, ensure_ascii=False)}\n\n"
        
        yield f"data: {json.dumps({'status': 'done'}, ensure_ascii=False)}\n\n"
