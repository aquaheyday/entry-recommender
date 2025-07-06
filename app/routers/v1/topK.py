from fastapi import APIRouter, Depends, HTTPException, Query, Path
from app.schemas.topK import TopKRequest, TopKResponse
from app.services.topK import get_recommendations_top_k
router = APIRouter(
    prefix="/v1",
    tags=["Recommendations"]
)

@router.get("/recommendations/top-k", response_model=TopKResponse)
def recommend(
    req: TopKRequest = Depends()
):
    return get_recommendations_top_k(
        tracking_key=req.tracking_key,
        lang=req.lang,
        top_k=req.top_k
    )
