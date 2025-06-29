from fastapi import APIRouter, HTTPException, Query
from app.services.trainer import train_site_model
from app.schemas.recommendation import TrainResponse

router = APIRouter(
    prefix="/v1",
    tags=["Training"]
)

@router.post(
    "/models",        # 기존 "/sites/{site_id}/models" 에서 변경
    response_model=TrainResponse
)
def train_user_model(
    tracking_key: str = Query(..., description="학습할 사이트의 ID")
):
    """
    POST /v1/models?site_id=6
    """
    try:
        return train_site_model(tracking_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
