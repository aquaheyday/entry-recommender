from fastapi import APIRouter, HTTPException
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services.recommender import get_recommendations

router = APIRouter(
    prefix="/recommend",
    tags=["Recommend"]
)

@router.post("/", response_model=RecommendationResponse)
def recommend_items(req: RecommendationRequest):
    try:
        return get_recommendations(req.site_id, req.top_k)
    except Exception as e:
        # 예외가 생기면 500으로
        raise HTTPException(status_code=500, detail=str(e))
