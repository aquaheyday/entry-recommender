import os
import numpy as np
import logging
import pickle
from functools import lru_cache
from fastapi import HTTPException
from core.model.lightfm_trainer import load_latest_model
from app.utils.model_utils import find_latest_version_dir
from app.schemas.topK import TopKResponse, TopKItem
from core.data_loader.clickhouse import load_popular_items, load_item_metadata_full

logger = logging.getLogger(__name__)

# 1) 모델 로드 캐시 (v{n}/{lang} 단위)
@lru_cache(maxsize=128)
def _load_model_cached(model_dir: str):
    logger.info(f"📦 캐시에서 모델 로딩: {model_dir}")
    return load_latest_model(model_dir)

# 1) 인기메타 캐시: tracking_key별로 한 번만 전체 메타를 dict로 로드
@lru_cache(maxsize=64)
def _load_full_meta_cached(tracking_key: str, lang: str | None = None) -> dict[str, dict]:
    """
    ClickHouse에서 tracking_key에 대한 전체 item metadata를 불러와
    {product_code: {...메타...}} 형태로 반환합니다.
    """
    df = load_item_metadata_full(tracking_key, lang=lang)
    df = df.fillna("")  # NaN 방지
    return df.set_index("product_code").to_dict(orient="index")

def fetch_popular_codes(
    tracking_key: str,
    lang: str = "und",
    top_k: int = 10
) -> list[str]:
    # 모델 기반 인기
    resp = get_model_popular_items(tracking_key, lang, top_k)
    return [item.product_code for item in resp.recommended_items]

def get_recommendations_top_k(
    tracking_key: str,
    lang: str = "und",
    top_k: int = 10
) -> TopKResponse:
    logger.info("✔️ 인기 상품 추천 로직 실행")

    # 1) 인기 코드 조회 (항상 List[str])
    codes = fetch_popular_codes(
        tracking_key,
        lang=lang,
        top_k=top_k
    )

    # 2) 전체 메타 딕셔너리 (기존 캐시)
    full_meta = _load_full_meta_cached(tracking_key, lang)

    # 3) RecommendationItem 생성
    items: list[TopKItem] = []
    for code in codes:
        meta = full_meta.get(code)
        if not meta:
            logger.warning(f"인기 추천 메타 없음: {code}")
            continue
        items.append(TopKItem(product_code=code, **meta))

    return TopKResponse(
        tracking_key=tracking_key,
        recommended_items=items
    )

def get_model_popular_items(
    tracking_key: str,
    lang: str = "und",
    top_k: int = 10
) -> TopKResponse:
    """
    LightFM 모델의 item_bias를 이용한 전역 인기 순위(top_k) 조회.
    """
    # 1) 최신 모델 디렉터리 찾기
    base = os.getenv("MODEL_BASE_DIR", "/app/models")
    try:
        model_dir = find_latest_version_dir(tracking_key, lang, base)
    except FileNotFoundError:
        raise HTTPException(404, "모델을 찾을 수 없습니다.")

    # 2) 모델·맵·메타 로드
    model, user_map, item_map, item_meta = _load_model_cached(model_dir)

    # 3) bias 배열에서 상위 top_k 인덱스 추출
    biases: np.ndarray = model.item_biases  # shape = (num_items,)
    k = min(top_k, biases.size)
    top_idxs = np.argpartition(-biases, k - 1)[:k]
    top_idxs = top_idxs[np.argsort(-biases[top_idxs])]

    # 4) 인덱스 → product_code
    codes = [item_map[idx] for idx in top_idxs]

    # 5) RecommendationItem 생성
    items = []
    for code in codes:
        meta = item_meta.get(code)
        if not meta:
            continue
        items.append(TopKItem(product_code=code, **meta))

    return TopKResponse(
        tracking_key=tracking_key,
        recommended_items=items
    )