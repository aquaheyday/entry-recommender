import os
import numpy as np
import logging
import pickle
from functools import lru_cache
from fastapi import HTTPException
from core.model.lightfm_trainer import load_latest_model
from app.utils.model_utils import find_latest_version_dir
from app.schemas.recommendation import RecommendationResponse, RecommendationItem
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
    df = load_item_metadata_full(tracking_key, lang)
    df = df.fillna("")  # NaN 방지
    return df.set_index("product_code").to_dict(orient="index")

def get_recommendations(
    tracking_key: str,
    anon_id: str,
    lang: str = "und",
    top_k: int = 10
) -> RecommendationResponse:
    """
    인기 상품 추천 (ClickHouse 집계 + cached 메타 조회)
    """
    logger.info("✔️ 인기 상품 추천 로직 실행")

    # 1) 인기 순위 조회 (product_code 리스트)
    df_pop = load_popular_items(tracking_key, top_k=top_k)
    codes: List[str] = df_pop["product_code"].tolist()

    # 2) 전체 메타 딕셔너리 (cached)
    full_meta = _load_full_meta_cached(tracking_key, lang)

    # 3) RecommendationItem 생성
    items: List[RecommendationItem] = []
    for code in codes:
        meta = full_meta.get(code)
        if not meta:
            logger.warning(f"인기추천 메타 없음: {code}")
            continue

        items.append(
            RecommendationItem(
                anon_id=anon_id,
                product_code=code,
                product_name=meta["product_name"],
                product_price=meta["product_price"],
                product_dc_price=meta["product_dc_price"],
                product_sold_out=meta["product_sold_out"],
                product_image_url=meta["product_image_url"],
                product_brand=meta["product_brand"],
                product_category_1_code=meta["product_category_1_code"],
                product_category_1_name=meta["product_category_1_name"],
                product_category_2_code=meta["product_category_2_code"],
                product_category_2_name=meta["product_category_2_name"],
                product_category_3_code=meta["product_category_3_code"],
                product_category_3_name=meta["product_category_3_name"],
                tracking_type=meta.get("tracking_type", ""),
                common_page_language=meta.get("common_page_language", ""),
                site_domain=meta["site_domain"],
                protocol=meta["protocol"],
            )
        )

    return RecommendationResponse(
        tracking_key=tracking_key,
        anon_id=anon_id,
        recommended_items=items
    )

def get_interest_based_recommendations(
    tracking_key: str,
    anon_id: str,
    lang: str = "und",
    top_k: int = 10
) -> RecommendationResponse:
    """
    학습된 LightFM 모델을 사용해 관심 기반 추천을 반환합니다.
    추천된 상품의 모든 메타(이름, 가격, 이미지 등)는
    학습 시 저장한 item_meta.pkl 에서 바로 가져옵니다.
    """
    base = os.getenv("MODEL_BASE_DIR", "/app/models")
    # 1) 최신 모델 디렉터리 찾기
    try:
        model_dir = find_latest_version_dir(tracking_key, lang, base)
    except FileNotFoundError:
        logger.warning("모델 디렉터리 없음 → 인기추천으로 폴백")
        return get_recommendations(tracking_key, anon_id, lang, top_k)

    # 2) 모델·맵·메타 로드 (LRU 캐시)
    try:
        model, user_map, item_map, item_meta = _load_model_cached(model_dir)
    except FileNotFoundError as e:
        logger.error(f"모델 파일 로드 실패: {e} → 인기추천 폴백")
        return get_recommendations(tracking_key, anon_id, lang, top_k)

    # 3) 사용자 존재 여부 체크
    if anon_id not in user_map:
        logger.info(f"{anon_id} 모델에 없음 → 인기추천 폴백")
        return get_recommendations(tracking_key, anon_id, lang, top_k)

    # 4) 예측: scores 계산
    uid = user_map[anon_id]
    ids = np.arange(len(item_map))
    scores = model.predict(uid, ids)

    # 5) 상위 k개 인덱스 추출
    k = min(top_k, scores.size)
    top_idxs = np.argpartition(-scores, k - 1)[:k]
    top_idxs = top_idxs[np.argsort(-scores[top_idxs])]

    # 6) 인덱스 → 상품코드
    rec_codes = [
        item_map[int(idx)]
        for idx in top_idxs
        if int(idx) in item_map
    ]
    logger.info(f"추천된 상품 코드: {rec_codes}")

    # 7) RecommendationItem 생성 (item_meta에서 바로 가져오기)
    items: list[RecommendationItem] = []
    for code in rec_codes:
        meta = item_meta.get(code)
        if not meta:
            logger.warning(f"메타없음: {code} (skip)")
            continue
        items.append(
            RecommendationItem(
                anon_id=anon_id,
                product_code=code,
                **meta
            )
        )

    # 8) 부족분은 인기추천으로 채우기
    if len(items) < top_k:
        pop_items = get_recommendations(tracking_key, anon_id, lang, top_k).recommended_items
        fill = [
            i for i in pop_items
            if i.product_code not in rec_codes
        ]
        items.extend(fill[: top_k - len(items)])

    # 9) 최종 반환
    return RecommendationResponse(
        tracking_key=tracking_key,
        anon_id=anon_id,
        recommended_items=items
    )