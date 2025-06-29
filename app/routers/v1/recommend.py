from fastapi import APIRouter, HTTPException, Query, Path
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services.recommender import (
    get_recommendations,
    get_interest_based_recommendations,
)

router = APIRouter(
    prefix="/v1/sites",
    tags=["Recommendations"]
)

@router.get("/{site_id}/recommendations/top", response_model=RecommendationResponse)
def get_top_recommendations(
    site_id: str,
    top_k: int = Query(
        10,  # 기본값 10
        ge=1,
        le=100,
        description="조회할 추천 개수 (1~100)"
    )
):
    try:
        return get_recommendations(site_id, top_k)
    except Exception as e:
        # 예외가 생기면 500으로
        raise HTTPException(status_code=500, detail=str(e))

# 1) 사용자 별 관심 기반 추천 (쿼리파라미터 방식)
@router.get(
    "/{site_id}/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
)
def get_user_interest_recommendations(
    site_id: str = Path(..., description="사이트 ID"),
    user_id: str = Path(..., description="사용자 ID"),
    top_k: int = Query(10, ge=1, le=100, description="조회할 추천 개수")
):
    try:
        return get_interest_based_recommendations(site_id, user_id, top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
