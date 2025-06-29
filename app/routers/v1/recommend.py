from fastapi import APIRouter, HTTPException, Query, Path
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services.recommender import (
    get_recommendations,
    get_interest_based_recommendations,
)

router = APIRouter(
    prefix="/v1",
    tags=["Recommendations"]
)

# 1) 사용자 별 관심 기반 추천 (쿼리파라미터 방식)
@router.get(
    "/recommendations",
    response_model=RecommendationResponse,
)
def get_user_interest_recommendations(
    tracking_key: str = Query(..., description="트래킹 key"),
    user_id: str = Query(..., description="사용자 ID"),
    top_k: int = Query(10, ge=1, le=100, description="조회할 추천 개수")
):
    try:
        return get_interest_based_recommendations(tracking_key, user_id, top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
