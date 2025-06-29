import os
import pickle
from fastapi import HTTPException
from core.data_loader.clickhouse import load_clickhouse_events
from core.preprocess.transformer import transform_interaction_matrix
from core.model.lightfm_trainer import train_model
from app.config import settings

def train_model_for_site(site_id: str) -> dict:
    """
    주어진 site_id에 대해 ClickHouse 이벤트를 불러와 행렬 변환 후 모델을 학습하고,
    models 디렉터리 아래에 버전별로 저장합니다.
    """
    # 데이터 로드 및 검증
    df = load_clickhouse_events(site_filter=site_id)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No events for site {site_id}")

    # 행렬 변환 및 모델 학습
    matrix, user_map, item_map = transform_interaction_matrix(df)
    model = train_model(matrix)

    # 저장 경로 설정 (기본 /app/models 또는 설정값)
    base_dir = settings.MODEL_BASE_DIR or "/app/models"
    site_dir = os.path.join(base_dir, f"site-{site_id}")
    os.makedirs(site_dir, exist_ok=True)

    # 버전 자동 관리: 기존 v* 폴더 중 최대 +1
    versions = [int(d[1:]) for d in os.listdir(site_dir)
                if d.startswith("v") and d[1:].isdigit()]
    next_ver = max(versions) + 1 if versions else 1
    version = f"v{next_ver}"
    version_dir = os.path.join(site_dir, version)
    os.makedirs(version_dir, exist_ok=True)

    # 파일 저장
    model_path = os.path.join(version_dir, "model.pkl")
    user_map_path = os.path.join(version_dir, "user_map.pkl")
    item_map_path = os.path.join(version_dir, "item_map.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(user_map_path, "wb") as f:
        pickle.dump(user_map, f)
    with open(item_map_path, "wb") as f:
        pickle.dump(item_map, f)

    return {
        "version": version,
        "model_path": model_path,
        "user_map_path": user_map_path,
        "item_map_path": item_map_path
    }
