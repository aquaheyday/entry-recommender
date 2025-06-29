from pydantic import BaseModel
from typing import List

class RecommendationRequest(BaseModel):
    site_id: str
    top_k: int = 10

class RecommendationResponse(BaseModel):
    site_id: str
    recommended_items: List[str]

class TrainResponse(BaseModel):
    version: str
    model_path: str
    user_map_path: str
    item_map_path: str
