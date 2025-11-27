# 다국어 번역 API

LangGraph와 OpenAI를 활용한 다국어 번역 서비스입니다.

## 주요 기능

- LangGraph 기반 번역 워크플로우
- 긴 텍스트 자동 청크 분할 (RecursiveCharacterTextSplitter)
- 병렬 번역 처리로 빠른 번역 속도
- OpenAI 스트리밍 지원
- 다양한 언어 지원

## 지원 언어

- 한국어 (ko)
- 일본어 (ja)
- 중국어 (zh)
- 베트남어 (vi)
- 프랑스어 (fr)
- 태국어 (th)
- 필리핀어 (tl)
- 크메르어 (km)

## 설치 방법

1. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 패키지 설치
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY를 설정하세요
```

## 실행 방법

```bash
python run.py
```

또는 uvicorn 직접 실행:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 실행되면 `http://localhost:8000`에서 접근할 수 있습니다.

API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

## API 사용 예제

### 기본 번역 API

```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "dalg_id": "test_id",
    "dalg_sn": "1",
    "user_indv_idntf_cd": "user123",
    "srch_task_se_cd": "TRANSLATE",
    "dmnd_msg": "안녕하세요. 반갑습니다.",
    "user_info": {
      "user_typ_cd": "USER",
      "user_typ_nm": "사용자",
      "user_lang_cd": "ko"
    },
    "service_info": {
      "lang_cd": "en"
    }
  }'
```

### 스트리밍 번역 API

```bash
curl -X POST "http://localhost:8000/translate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "dalg_id": "test_id",
    "dalg_sn": "1",
    "user_indv_idntf_cd": "user123",
    "srch_task_se_cd": "TRANSLATE",
    "dmnd_msg": "긴 텍스트를 번역합니다...",
    "user_info": {
      "user_typ_cd": "USER",
      "user_typ_nm": "사용자",
      "user_lang_cd": "ko"
    },
    "service_info": {
      "lang_cd": "ja"
    }
  }'
```

## 프로젝트 구조

```
.
├── api/                    # FastAPI 관련
│   ├── __init__.py
│   └── main.py            # FastAPI 메인 애플리케이션
├── schema/                 # Pydantic 스키마 정의
│   ├── __init__.py
│   └── schemas.py         # API 요청/응답 모델
├── state/                  # 상태 정의
│   ├── __init__.py
│   └── translation_state.py  # LangGraph 상태 정의
├── prompt/                 # 프롬프트 관리
│   ├── __init__.py
│   └── prompts.py         # 번역 프롬프트 생성
├── generator/              # LLM 생성기
│   ├── __init__.py
│   └── llm_generator.py   # LLM 호출 및 번역 로직
├── graph/                  # LangGraph 정의
│   ├── __init__.py
│   └── translation_graph.py  # 번역 그래프 구성
├── orchestrator/           # 오케스트레이터
│   ├── __init__.py
│   └── translation_orchestrator.py  # 번역 프로세스 조율
├── utils/                  # 유틸리티
│   ├── __init__.py
│   └── language_utils.py  # 언어 관련 유틸리티
├── run.py                 # 서버 실행 스크립트
├── requirements.txt       # 패키지 의존성
├── env.example           # 환경 변수 예제
└── README.md             # 프로젝트 문서
```

## 번역 프로세스

1. **텍스트 분할**: 긴 텍스트는 RecursiveCharacterTextSplitter를 사용하여 적절한 크기의 청크로 분할합니다.
2. **병렬 번역**: 분할된 청크들을 병렬로 번역하여 속도를 향상시킵니다.
3. **결과 병합**: 번역된 청크들을 순서대로 합쳐 최종 번역 결과를 생성합니다.

## 주의사항

- OpenAI API 키가 필요합니다.
- API 사용량에 따라 비용이 발생할 수 있습니다.
- 긴 텍스트는 자동으로 청크로 분할되어 처리됩니다.

## 라이선스

이 프로젝트는 개인 프로젝트입니다.

