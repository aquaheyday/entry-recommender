import os
import pickle
import pandas as pd
from fastapi import HTTPException
from core.data_loader.clickhouse import load_clickhouse_events
from core.preprocess.transformer import transform_interaction_matrix
from core.model.lightfm_trainer import train_model
from app.config import settings

def train_models_for_site(tracking_key: str) -> dict:
    """
    1) ClickHouse에서 events + 메타 컬럼이 모두 포함된 DataFrame을 한 번만 불러옵니다.
    2) Pandas로 인터랙션 매트릭스와 상품 메타를 분리합니다.
    3) 언어별로 LightFM 모델을 학습하고, 모델·맵·메타를 저장합니다.
    """
    # 1) 전체 이벤트 + 메타 한 번에 로드
    df = load_clickhouse_events(tracking_filter=tracking_key)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No events for site {tracking_key}")

    # ◀ common_ts를 datetime으로 변환
    df['common_ts'] = pd.to_datetime(df['common_ts'])

    # --- 상품 메타를 한 번만 뽑아서 dict로 저장 ---
    # ① common_ts 내림차순 정렬 → 같은 product_code 중 맨 앞 행(=최신)만 남기기
    meta_df = (
        df
        .sort_values(['product_code', 'common_ts'], ascending=[True, False])
        .drop_duplicates(subset=["product_code"], keep='first')
    )

    base_dir = settings.MODEL_BASE_DIR or "/app/models/lightfm"
    site_root = os.path.join(base_dir, f"{tracking_key}")
    os.makedirs(site_root, exist_ok=True)

    # 버전 관리
    versions = [int(d[1:]) for d in os.listdir(site_root) if d.startswith("v") and d[1:].isdigit()]
    next_ver = max(versions) + 1 if versions else 1
    version = f"v{next_ver}"
    version_dir = os.path.join(site_root, version)
    os.makedirs(version_dir, exist_ok=True)

    results = {}

    # --- 상품 메타를 한 번만 뽑아서 dict로 저장 ---
    # drop_duplicates로 product_code별 첫 행만 남기고 to_dict
    meta_dict = (
        meta_df
        .drop_duplicates(subset=["product_code"])
        .set_index("product_code")[
            [
                "product_name",
                "product_price",
                "product_dc_price",
                "product_sold_out",
                "product_image_url",
                "product_brand",
                "product_category_1_code",
                "product_category_1_name",
                "product_category_2_code",
                "product_category_2_name",
                "product_category_3_code",
                "product_category_3_name",
                "product_url",
                "tracking_type",
                "common_page_language",
            ]
        ]
        .fillna("")  # NaN 방지
        .to_dict(orient="index")
    )

    # 2) 언어별 그룹핑 & 학습
    for lang, group_df in df.groupby("common_page_language"):
        if group_df.empty:
            continue

        # ▶ 언어별 메타만 뽑아서 정렬·중복제거
        lang_meta_df = (
            group_df
            .sort_values(['product_code', 'common_ts'], ascending=[True, False])
            .drop_duplicates(subset=["product_code"], keep='first')
        )
        lang_meta_dict = lang_meta_df.set_index("product_code")[
            [
                "product_name", "product_price", "product_dc_price",
                "product_sold_out", "product_image_url", "product_brand",
                "product_category_1_code", "product_category_1_name",
                "product_category_2_code", "product_category_2_name",
                "product_category_3_code", "product_category_3_name",
                "product_url",
                "tracking_type", "common_page_language",
            ]
        ].fillna("").to_dict(orient="index")

        # 2-1) interaction matrix 변환
        matrix, user_map, item_map = transform_interaction_matrix(group_df)

        # 2-2) 모델 학습
        model = train_model(matrix)

        # 2-3) 언어별 디렉터리
        lang_dir = os.path.join(version_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)

        # 헬퍼: pickle 저장
        def _save(obj, filename):
            with open(os.path.join(lang_dir, filename), "wb") as f:
                pickle.dump(obj, f)

        # 2-4) 모델 및 맵 저장
        _save(model,      "model.pkl")
        _save(user_map,   "user_map.pkl")
        _save(item_map,   "item_map.pkl")

        # 2-5) 메타 저장 (item_map에 있는 코드만 필터)
        filtered_meta = {
            code: lang_meta_dict[code]
            for code in item_map.keys()
            if code in lang_meta_dict
        }
        _save(filtered_meta, "item_meta.pkl")

        results[lang] = {
            "version":        version,
            "model_path":     os.path.join(lang_dir, "model.pkl"),
            "user_map_path":  os.path.join(lang_dir, "user_map.pkl"),
            "item_map_path":  os.path.join(lang_dir, "item_map.pkl"),
            "item_meta_path": os.path.join(lang_dir, "item_meta.pkl"),
        }

    return results
