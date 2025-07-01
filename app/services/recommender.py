import os
import numpy as np
import pandas as pd
import logging
from fastapi import HTTPException
from core.data_loader.clickhouse import (
    load_clickhouse_events,
    load_clickhouse_item_metadata
)
from core.model.lightfm_trainer import load_latest_model
from core.utils.model_path import find_latest_version_dir
from app.schemas.recommendation import RecommendationResponse

logger = logging.getLogger(__name__)

def get_recommendations(tracking_key: str, top_k: int) -> RecommendationResponse:
    """
    사이트 전체 이벤트로부터 인기 상품 top_k 리스트를 반환합니다.
    """
    df = load_clickhouse_events(tracking_filter=tracking_key)
    if df is None or df.empty:
        logger.info("No events for site %s, returning empty list", tracking_key)
        return RecommendationResponse(
            tracking_key=tracking_key,
            recommended_items=[]
        )

    counts = df["product_code"].value_counts()
    top_items = counts.head(top_k).index.tolist()
    logger.info("Top %d popular items for site %s: %s", top_k, tracking_key, top_items)

    return RecommendationResponse(
        tracking_key=tracking_key,
        recommended_items=top_items
    )


def get_interest_based_recommendations(
    tracking_key: str,
    anon_id: str,
    lang: str,
    top_k: int
) -> RecommendationResponse:
    """
    사용자별 이벤트와 관심 카테고리, 언어별 학습된 모델을 이용해 추천 목록을 반환합니다.
    """
    df = load_clickhouse_events(tracking_filter=tracking_key)
    if df is None or df.empty:
        logger.info("No events for site %s, returning empty list", tracking_key)
        return RecommendationResponse(
            tracking_key=tracking_key,
            recommended_items=[]
        )

    if "page_language" in df.columns:
        df = df[df["page_language"] == lang]

    df_user = df[df["anon_id"] == anon_id]
    if df_user.empty:
        return get_recommendations(tracking_key, top_k)

    meta = load_clickhouse_item_metadata(tracking_filter=tracking_key)
    df_user = df_user.merge(meta, left_on="product_code", right_on="item_id", how="left")

    if "category_1" in df_user.columns:
        cat_col = "category_1"
    elif "category" in df_user.columns:
        cat_col = "category"
    else:
        raise HTTPException(status_code=500, detail="Category column not found in metadata")

    top_categories = (
        df_user[cat_col]
        .value_counts()
        .nlargest(3)
        .index
        .tolist()
    )

    # 4) 언어별 모델 버전 디렉터리 찾기
    base_dir = os.getenv("MODEL_BASE_DIR", "/app/models")
    try:
        version_dir = find_latest_version_dir(
            site_id=tracking_key,
            lang=lang,
            base_dir=base_dir
        )
    except FileNotFoundError as e:
        logger.warning("Model directory not found: %s", e)
        return get_recommendations(tracking_key, top_k)

    # 5) 최신 모델 로드
    model, user_map, item_map = load_latest_model(version_dir)

    if anon_id not in user_map:
        raise HTTPException(status_code=404, detail=f"User {anon_id} not in model for lang {lang}")

    user_idx = user_map[anon_id]
    n_items = len(item_map)

    scores = model.predict(user_idx, np.arange(n_items))
    df_scores = pd.DataFrame({
        "item_id": list(item_map.values()),
        "score": scores
    }).merge(meta, on="item_id", how="left")

    viewed = set(df_user["product_code"])
    df_filtered = df_scores[
        df_scores[cat_col].isin(top_categories) &
        (~df_scores["item_id"].isin(viewed))
    ]

    top_items = df_filtered.nlargest(top_k, "score")["item_id"].tolist()
    return RecommendationResponse(
        tracking_key=tracking_key,
        recommended_items=top_items
    )