import os
import pickle
from fastapi import HTTPException
from core.data_loader.clickhouse import load_clickhouse_events
from core.preprocess.transformer import transform_interaction_matrix
from core.model.lightfm_trainer import train_model
from app.config import settings

def train_models_for_site(tracking_key: str) -> dict:
    """
    tracking_key 에 해당하는 이벤트를 ClickHouse 에서 가져와,
    page_language 기준으로 그룹핑 후 언어별 LightFM 모델을 학습·저장합니다.
    저장 경로는 {tracking_key}/v{version}/{lang}/ 입니다.
    """
    # 1. 전체 이벤트 로드
    df = load_clickhouse_events(tracking_filter=tracking_key)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No events for site {tracking_key}")

    # 2. 저장 디렉토리 준비
    base_dir = settings.MODEL_BASE_DIR or "/app/models"
    site_root = os.path.join(base_dir, f"lightfm/{tracking_key}")
    os.makedirs(site_root, exist_ok=True)

    # 3. 버전 자동 관리: 기존 v* 폴더 중 최대 +1
    versions = [int(d[1:]) for d in os.listdir(site_root)
                if d.startswith("v") and d[1:].isdigit()]
    next_ver = max(versions) + 1 if versions else 1
    version = f"v{next_ver}"
    version_dir = os.path.join(site_root, version)
    os.makedirs(version_dir, exist_ok=True)

    results = {}

    # 4. 언어별 그룹핑 & 학습 반복
    for lang, group_df in df.groupby("common_page_language"):
        if group_df.empty:
            continue

        # 4-1) interaction matrix 변환
        matrix, user_map, item_map = transform_interaction_matrix(group_df)

        # 4-2) 모델 학습
        model = train_model(matrix)

        # 4-3) 언어별 디렉토리 생성
        lang_dir = os.path.join(version_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)

        # 4-4) 파일로 저장
        def _save(obj, filename):
            with open(os.path.join(lang_dir, filename), "wb") as f:
                pickle.dump(obj, f)

        _save(model, "model.pkl")
        _save(user_map, "user_map.pkl")
        _save(item_map, "item_map.pkl")

        results[lang] = {
            "version": version,
            "model_path": os.path.join(lang_dir, "model.pkl"),
            "user_map_path": os.path.join(lang_dir, "user_map.pkl"),
            "item_map_path": os.path.join(lang_dir, "item_map.pkl"),
        }

    return results
