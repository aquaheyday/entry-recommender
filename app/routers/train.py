from fastapi import APIRouter, HTTPException
from app.services.trainer import train_site_model
from ..schemas.recommendation import TrainResponse

router = APIRouter(
    prefix="/train",
    tags=["Training"]
)

@router.post("/{site_id}", response_model=TrainResponse)
def train_user_model(site_id: str):
    try:
        return train_site_model(site_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
