"""번역 프롬프트 생성"""


def create_translation_prompt(target_lang: str, target_lang_name: str) -> str:
    """번역 프롬프트 생성"""
    return f"""당신은 전문 번역가입니다. 다음 텍스트를 {target_lang_name}({target_lang})로 정확하게 번역해주세요.

번역 규칙:
1. 원문의 의미와 톤을 정확하게 유지하세요
2. 자연스러운 {target_lang_name} 표현을 사용하세요
3. 전문 용어는 적절하게 번역하세요
4. 문맥을 고려하여 번역하세요

번역할 텍스트:"""

