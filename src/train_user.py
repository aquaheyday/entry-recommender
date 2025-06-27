import os
import pickle
from src.data_loader.clickhouse import load_clickhouse_events
from src.preprocess.transformer import transform_interaction_matrix
from src.model.lightfm_trainer import train_model

def next_version_dir(base_path: str) -> str:

    os.makedirs(base_path, exist_ok=True)
    versions = []
    for name in os.listdir(base_path):
        if name.startswith("v") and name[1:].isdigit():
            versions.append(int(name[1:]))
    next_ver = max(versions, default=0) + 1
    return f"v{next_ver}"

def train_model_for_site(site_id: str) -> dict:
    print(f"🚀 모델 학습 시작: site_id={site_id}")
    df = load_clickhouse_events(site_filter=site_id)
    if df.empty:
        raise ValueError(f"❌ 사용자 {site_id}에 대한 데이터가 없습니다.")

    matrix, user_map, item_map = transform_interaction_matrix(df)
    model = train_model(matrix)

    # 버전 디렉터리 결정
    site_base = f"models/site-{site_id}"
    version_dir = next_version_dir(site_base)
    full_dir = os.path.join(site_base, version_dir)
    os.makedirs(full_dir, exist_ok=True)

    # 파일 경로
    model_path     = os.path.join(full_dir, f"model.pkl")
    user_map_path  = os.path.join(full_dir, f"user_map.pkl")
    item_map_path  = os.path.join(full_dir, f"item_map.pkl")

    # 저장
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(user_map_path, "wb") as f:
        pickle.dump(user_map, f)
    with open(item_map_path, "wb") as f:
        pickle.dump(item_map, f)

    print(f"✅ 모델 저장 완료: {model_path}")
    return {
        "version": version_dir,
        "model_path": model_path,
        "user_map_path": user_map_path,
        "item_map_path": item_map_path
    }
