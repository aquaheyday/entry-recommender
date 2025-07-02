from pydantic import BaseModel
from typing import List

class RecommendationRequest(BaseModel):
    tracking_key: str
    top_k: int = 10

class TrainResponse(BaseModel):
    version: str
    model_path: str
    user_map_path: str
    item_map_path: str

class RecommendationItem(BaseModel):
    product_code: str
    product_name: str
    product_price: float
    product_dc_price: float
    product_sold_out: bool
    product_image_url: str
    product_brand: str
    product_category_1_code: str
    product_category_1_name: str
    product_category_2_code: str
    product_category_2_name: str
    product_category_3_code: str
    product_category_3_name: str
    site_domain: str
    protocol: str

class RecommendationResponse(BaseModel):
    tracking_key: str
    anon_id: str
    recommended_items: List[RecommendationItem]
