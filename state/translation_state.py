"""번역 상태 정의"""
from typing import List, TypedDict


class TranslationState(TypedDict):
    """번역 상태를 관리하는 클래스"""
    original_text: str
    target_lang_cd: str
    target_lang_name: str
    chunks: List[str]
    translated_chunks: List[str]
    final_translation: str
    chunk_count: int

