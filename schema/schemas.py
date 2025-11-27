"""API 요청/응답 스키마 정의"""
from pydantic import BaseModel


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

