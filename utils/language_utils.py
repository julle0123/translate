"""언어 관련 유틸리티 함수"""

# 언어 코드와 언어 이름 매핑
LANGUAGE_MAP = {
    "ko": "한국어",
    "en": "영어",
    "ja": "일본어",
    "zh": "중국어",
    "vi": "베트남어",
    "ru": "러시아어",
    "th": "태국어",
    "fr": "프랑스어",
    "mn": "몽골어",
    "km": "크메르어",
    "tl": "필리핀어",
}


def get_language_name(lang_cd: str) -> str:
    """언어 코드를 언어 이름으로 변환"""
    return LANGUAGE_MAP.get(lang_cd, lang_cd)

