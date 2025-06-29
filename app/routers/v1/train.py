from fastapi import APIRouter, HTTPException, Path
from app.services.trainer import train_site_model
from app.schemas.recommendation import TrainResponse

router = APIRouter(
    prefix="/v1",
    tags=["Training"]
)

@router.post("/sites/{site_id}/models", response_model=TrainResponse)
def train_user_model(site_id: str):
    try:
        return train_site_model(site_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
