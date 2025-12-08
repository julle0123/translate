"""FastAPI 메인 애플리케이션"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import json

import os
from schema.schemas import TranslationRequest
from generator.translation_generator import TranslationGenerator
from langchain_openai import ChatOpenAI
from utils.language_utils import get_language_name

# 환경 변수 로드
load_dotenv()

# Generator 인스턴스 생성
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.3,
    streaming=True,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
generator = TranslationGenerator(llm=llm)

app = FastAPI(
    title="Translation API",
    description="LangGraph 기반 다국어 번역 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Translation API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"}


@app.post("/translate")
async def translate(request: TranslationRequest):
    """
    텍스트 번역 API
    
    Request Body:
    - dalg_id: 대화 ID
    - dalg_sn: 대화 순번
    - user_indv_idntf_cd: 사용자 개인 식별 코드
    - srch_task_se_cd: 검색 작업 구분 코드
    - dmnd_msg: 요청 메시지 (번역할 텍스트)
    - user_info: 사용자 정보 (user_typ_cd, user_typ_nm, user_lang_cd)
    - service_info: 서비스 정보 (lang_cd: 번역 대상 언어 코드)
    """
    try:
        # 번역 대상 언어 코드 확인
        target_lang_cd = request.service_info.lang_cd
        
        if not target_lang_cd:
            raise HTTPException(
                status_code=400,
                detail="service_info.lang_cd is required"
            )
        
        if not request.dmnd_msg:
            raise HTTPException(
                status_code=400,
                detail="dmnd_msg is required"
            )
        
        # Generator를 직접 호출하여 번역 실행
        target_lang_name = get_language_name(target_lang_cd)
        result = await generator.translate_text_complete(
            text=request.dmnd_msg,
            target_lang_cd=target_lang_cd,
            target_lang_name=target_lang_name
        )
        translated_text = result["final_translation"]
        
        return {
            "dalg_id": request.dalg_id,
            "dalg_sn": request.dalg_sn,
            "user_indv_idntf_cd": request.user_indv_idntf_cd,
            "srch_task_se_cd": request.srch_task_se_cd,
            "original_msg": request.dmnd_msg,
            "translated_msg": translated_text,
            "target_lang_cd": target_lang_cd,
            "user_info": {
                "user_typ_cd": request.user_info.user_typ_cd,
                "user_typ_nm": request.user_info.user_typ_nm,
                "user_lang_cd": request.user_info.user_lang_cd
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


@app.post("/translate/stream")
async def translate_stream(request: TranslationRequest):
    """
    스트리밍 방식 텍스트 번역 API
    
    동일한 Request Body를 사용하지만, 스트리밍 방식으로 결과를 반환합니다.
    """
    try:
        # 번역 대상 언어 코드 확인
        target_lang_cd = request.service_info.lang_cd
        
        if not target_lang_cd:
            raise HTTPException(
                status_code=400,
                detail="service_info.lang_cd is required"
            )
        
        if not request.dmnd_msg:
            raise HTTPException(
                status_code=400,
                detail="dmnd_msg is required"
            )
        
        # 스트리밍 번역 실행 (orchestrator 사용)
        from orchestrator.translation_orchestrator import TranslationOrchestrator
        
        # service_info 설정 (lang_cd는 node에서 처리)
        service_info = {"lang_cd": target_lang_cd}
        orchestrator = TranslationOrchestrator(service_info=service_info)
        
        async def generate():
            import time
            start_time = time.time()
            first_token_time = None
            token_count = 0
            
            async for chunk in orchestrator.run(message=request.dmnd_msg):
                token_count += 1
                
                # 첫 토큰 시간 측정
                if first_token_time is None and token_count == 1:
                    first_token_time = time.time()
                    elapsed = first_token_time - start_time
                    print(f"⚡ [첫 토큰] {elapsed:.3f}초 소요 (번역 시작)")
                
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. Please set it in .env file.")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

