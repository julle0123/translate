"""API 요청/응답 스키마 정의"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any


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


class DataResponse(BaseModel):
    """DataResponse 기본 클래스"""
    model_config = ConfigDict(populate_by_name=True)
    
    index: int = Field(default=0, alias="index")
    step: str = Field(default="message", alias="step")
    rspns_msg: str = Field(default="", alias="answer")
    cmptn_yn: bool = Field(default=False, alias="completion")
    tokn_info: Dict[str, Any] = Field(default_factory=dict, alias="token_info")
    link_info: List[Dict[str, Any]] = Field(default_factory=list, alias="link_info")
    src_doc_info: List[Dict[str, Any]] = Field(default_factory=list, alias="src_doc_info")


class SSEChunk(DataResponse):
    """응답 type별로 chunk를 포맷팅하는 함수"""
    
    def to_msg(self) -> str:
        event = {
            "init": "message",
            "status": "status",
            "delta": "message",
            "final": "done",
            "error": "error",
        }.get(self.step, "message")
        msg_str = f"data: {self.model_dump_json(by_alias=False)}\n\n"
        return msg_str.encode("utf-8")

