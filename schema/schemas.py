"""API 요청/응답 스키마 정의"""
from pydantic import BaseModel
from typing import Optional


class UserInfo(BaseModel):
    user_typ_cd: str
    user_typ_nm: str
    user_lang_cd: str


class ServiceInfo(BaseModel):
    lang_cd: str


class TranslationRequest(BaseModel):
    dalg_id: str
    dalg_sn: str
    user_indv_idntf_cd: str
    srch_task_se_cd: str
    dmnd_msg: str
    user_info: UserInfo
    service_info: ServiceInfo


class SSEChunk(BaseModel):
    """SSE 스트리밍용 청크 클래스"""
    step: str  # init, status, delta, final, error
    answer: Optional[str] = None
    completion: Optional[bool] = None
    
    def to_msg(self) -> bytes:
        """SSE 형식의 메시지로 변환"""
        event = {
            "init": "message",
            "status": "status",
            "delta": "message",
            "final": "done",
            "error": "error",
        }.get(self.step, "message")
        
        msg_str = f"data: {self.model_dump_json(by_alias=False)}\n\n"
        return msg_str.encode("utf-8")

