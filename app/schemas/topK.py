from pydantic import BaseModel, Field
from typing import List, Literal

class TopKRequest(BaseModel):
    tracking_key: str = Field(..., description="사이트 고유 트래킹 키")
    lang: str = Field("und", description="페이지 언어 코드 (default und)")
    top_k: int = Field(
        10,
        ge=1, le=100,
        description="추천할 아이템 개수 (1~100)"
    )

class TopKItem(BaseModel):
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
    product_url: str

class TopKResponse(BaseModel):
    tracking_key: str
    recommended_items: List[TopKItem]