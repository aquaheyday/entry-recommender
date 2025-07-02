from typing import Dict
from pydantic import BaseModel

class ModelInfo(BaseModel):
    model_path: str
    user_map_path: str
    item_map_path: str
    item_meta_path: str

class TrainResponse(BaseModel):
    tracking_key: str
    models: Dict[str, ModelInfo]   # 언어 코드 → ModelInfo 매핑
