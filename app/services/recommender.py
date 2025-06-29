import os
import pickle
import numpy as np
import pandas as pd
import logging
from fastapi import HTTPException
from core.data_loader.clickhouse import (
    load_clickhouse_events,
    load_clickhouse_item_metadata
)
from core.preprocess.transformer import transform_interaction_matrix
from core.model.lightfm_trainer import load_latest_model
from app.schemas.recommendation import RecommendationResponse

logger = logging.getLogger(__name__)

def get_recommendations(tracking_key: str, top_k: int):
    """
    사이트 전체 이벤트로부터 인기 상품 top_k 리스트를 반환합니다.
    """
    df = load_clickhouse_events(tracking_filter=tracking_key)
    if df is None or df.empty:
        logger.info("No events for site %s, returning empty list", tracking_key)
        return {"tracking_key": tracking_key, "recommended_items": []}

    # 상품코드별 빈도 집계
    counts = df["product_code"].value_counts()
    top_items = counts.head(top_k).index.tolist()
    logger.info("Top %d popular items for site %s: %s", top_k, tracking_key, top_items)

    return {"tracking_key": tracking_key, "recommended_items": top_items}

def get_interest_based_recommendations(
    tracking_key: str,
    user_id: str,
    top_k: int
) -> RecommendationResponse:
    # 1) 사이트 전체 이벤트 로드 + 사용자 필터링
    df = load_clickhouse_events(tracking_filter=tracking_key)
    df_user = df[df["anon_id"] == user_id]

    # 폴백: 히스토리 없으면 전체 추천으로 대체
    if df_user.empty:
        return get_recommendations(tracking_key, top_k)

    # 2) 카테고리 메타데이터 로드 & 병합
    meta = load_clickhouse_item_metadata(tracking_filter=tracking_key)
    df_user = df_user.merge(meta, on="item_id", how="left")

    # 상위 3개 카테고리 추출
    top_categories = (
        df_user["category"]
        .value_counts()
        .nlargest(3)
        .index
        .tolist()
    )

    # 3) 최신 모델 로드
    model_dir = os.path.join(os.getenv("MODEL_BASE_DIR", "/app/models"), f"{tracking_key}")
    model, user_map, item_map = load_latest_model(model_dir)

    if user_id not in user_map:
        raise HTTPException(status_code=404, detail=f"User {user_id} not in model")

    user_idx = user_map[user_id]
    n_items = len(item_map)

    # 4) 모든 아이템 점수 예측
    scores = model.predict(user_idx, np.arange(n_items))
    df_scores = pd.DataFrame({
        "item_id": list(item_map.values()),  # load_latest_model 에서 idx->item_id 맵을 반환했다면 .values()
        "score": scores
    }).merge(meta, on="item_id", how="left")

    # 5) 관심 카테고리 필터 & 이미 본 아이템 제외
    viewed = set(df_user["item_id"])
    df_filtered = df_scores[
        df_scores["category"].isin(top_categories) &
        (~df_scores["item_id"].isin(viewed))
    ]

    # 6) top_k 선택 및 반환
    top_items = df_filtered.nlargest(top_k, "score")["item_id"].tolist()
    return RecommendationResponse(items=top_items)