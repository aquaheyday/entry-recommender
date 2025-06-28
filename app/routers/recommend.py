from fastapi import APIRouter, HTTPException
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from ..services.recommender import get_recommendations

router = APIRouter()

@router.post("/", response_model=RecommendationResponse)
def recommend_items(req: RecommendationRequest):
    try:
        return get_recommendations(req.site_id, req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
