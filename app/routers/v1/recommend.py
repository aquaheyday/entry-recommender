from fastapi import APIRouter, HTTPException, Query, Path
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services.recommender import (
    get_recommendations,
    get_interest_based_recommendations,
)
import logging
router = APIRouter(
    prefix="/v1",
    tags=["Recommendations"]
)

logger = logging.getLogger(__name__)

@router.get("/recommendations", response_model=RecommendationResponse)
def recommend(
    tracking_key: str = Query(...),
    anon_id: str = Query(...),
    lang: str = Query("und"),
    top_k: int = Query(10, ge=1, le=100),
):
    try:
        # 관심 기반 추천
        return get_interest_based_recommendations(tracking_key, anon_id, lang, top_k)
    except Exception as e:
        # 폴백: 인기 추천
        return get_recommendations(tracking_key, anon_id, lang, top_k)
