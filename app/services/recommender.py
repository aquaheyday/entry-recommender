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
from core.model.lightfm_trainer import load_latest_model
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

    # 상품코드별 빈도 집계
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
    # 1) 사이트 전체 이벤트 로드
    df = load_clickhouse_events(tracking_filter=tracking_key)
    if df is None or df.empty:
        logger.info("No events for site %s, returning empty list", tracking_key)
        return RecommendationResponse(
            tracking_key=tracking_key,
            recommended_items=[]
        )

    # 1.1) 언어 필터 적용 (page_language 컬럼 기준)
    if "page_language" in df.columns:
        df = df[df["page_language"] == lang]

    # 1.2) 사용자 필터링
    df_user = df[df["anon_id"] == anon_id]

    # 폴백: 히스토리 없으면 전체 추천으로 대체
    if df_user.empty:
        return get_recommendations(tracking_key, top_k)

    # 2) 카테고리 메타데이터 로드 & 병합 (product_code -> item_id)
    meta = load_clickhouse_item_metadata(tracking_filter=tracking_key)
    df_user = df_user.merge(meta, left_on="product_code", right_on="item_id", how="left")

    # 3) 상위 3개 카테고리 추출 (meta 컬럼명에 따라 수정)
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

    # 4) 언어별 모델 디렉토리 경로 설정
    base_dir = os.getenv("MODEL_BASE_DIR", "/app/models")
    model_root = os.path.join(base_dir, f"lightfm/{tracking_key}/{lang}")

    # 언어별 모델이 없으면 und(미설정) 디렉토리로 폴백
    if not os.path.isdir(model_root):
        fallback_root = os.path.join(base_dir, f"lightfm/{tracking_key}/und")
        logger.warning("Model directory for lang '%s' not found, falling back to 'und'", lang)
        model_root = fallback_root

    # 5) 최신 모델 로드
    model, user_map, item_map = load_latest_model(model_root)

    # 6) 사용자 인덱스 확인
    if anon_id not in user_map:
        raise HTTPException(status_code=404, detail=f"User {anon_id} not in model for lang {lang}")

    user_idx = user_map[anon_id]
    n_items = len(item_map)

    # 7) 모든 아이템 점수 예측
    scores = model.predict(user_idx, np.arange(n_items))
    df_scores = pd.DataFrame({
        "item_id": list(item_map.values()),
        "score": scores
    })
    # 메타데이터 merge 역시 item_id 기준으로
    df_scores = df_scores.merge(meta, on="item_id", how="left")

    # 8) 관심 카테고리 필터 & 이미 본 아이템 제외
    viewed = set(df_user["product_code"])
    df_filtered = df_scores[
        df_scores[cat_col].isin(top_categories) &
        (~df_scores["item_id"].isin(viewed))
    ]

    # 9) top_k 선택 및 반환
    top_items = df_filtered.nlargest(top_k, "score")["item_id"].tolist()
    return RecommendationResponse(
        tracking_key=tracking_key,
        recommended_items=top_items
    )