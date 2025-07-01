from fastapi import APIRouter
from app.schemas.train import TrainResponse
from app.services.trainer import train_models_for_site

router = APIRouter(
    prefix="/v1",
    tags=["Training"]
)

@router.post(
    "/sites/{tracking_key}/models",
    response_model=TrainResponse,
)
async def train_site_model_endpoint(tracking_key: str):
    """
    언어별로 학습된 모델을 돌려줍니다.
    """
    results = train_models_for_site(tracking_key)
    # results 예시: { "ko": {...}, "en": {...}, "und": {...} }
    return {
        "tracking_key": tracking_key,
        "models": results
    }
