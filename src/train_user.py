import os
import pickle
from src.data_loader.clickhouse import load_clickhouse_events
from src.preprocess.transformer import transform_interaction_matrix
from src.model.lightfm_trainer import train_model


def train_model_for_site(site_id: str) -> dict:
    print(f"🚀 모델 학습 시작: site_id={site_id}")
    df = load_clickhouse_events(site_filter=site_id)

    if df.empty:
        raise ValueError(f"❌ 사용자 {site_id}에 대한 데이터가 없습니다.")

    matrix, user_map, item_map = transform_interaction_matrix(df)
    model = train_model(matrix)

    os.makedirs("models/user_models", exist_ok=True)

    model_path = f"models/user_models/model_{site_id}.pkl"
    user_map_path = f"models/user_models/user_map_{site_id}.pkl"
    item_map_path = f"models/user_models/item_map_{site_id}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(user_map_path, "wb") as f:
        pickle.dump(user_map, f)

    with open(item_map_path, "wb") as f:
        pickle.dump(item_map, f)

    print(f"✅ 모델 저장 완료: {model_path}")
    return {
        "model_path": model_path,
        "user_map_path": user_map_path,
        "item_map_path": item_map_path
    }
