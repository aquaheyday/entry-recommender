import os, pickle
from .data_loader.clickhouse import load_clickhouse_events
from .preprocess.transformer import transform_interaction_matrix
from .model.lightfm_trainer import train_model

def train_model_for_site(site_id: str) -> dict:
    df = load_clickhouse_events(site_filter=site_id)
    matrix, user_map, item_map = transform_interaction_matrix(df)
    model = train_model(matrix)

    base = f"models/site-{site_id}"
    os.makedirs(base, exist_ok=True)
    # 버전 디렉터리 로직 생략: v1 etc
    path = f"{base}/model.pkl"
    pickle.dump(model, open(path, "wb"))
    # user_map, item_map 저장...
    return {"version": "v1", "model_path": path, "user_map_path": "", "item_map_path": ""}
