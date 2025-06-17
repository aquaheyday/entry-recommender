from fastapi import FastAPI
from src.model.recommender import recommend
from src.utils.model_utils import load_model_and_mappings
from src.data_loader.clickhouse import load_clickhouse_events
from src.preprocess.transformer import transform_interaction_matrix
import os

app = FastAPI()

model = user_map = item_map = df = matrix = None

@app.on_event("startup")
def startup_event():
    global model, user_map, item_map, df, matrix
    if not os.path.exists("models/lightfm_model.pkl"):
        print("⚠️ 모델이 존재하지 않습니다. 먼저 학습을 수행하세요.")
        return
    model, user_map, item_map = load_model_and_mappings()
    df = load_clickhouse_events()
    matrix, _, _ = transform_interaction_matrix(df)

@app.get("/recommend")
def get_recommendations(user_id: str, top_n: int = 10):
    if model is None:
        return {"error": "모델이 로드되지 않았습니다. 먼저 모델을 학습하세요."}
    result = recommend(model, user_id, matrix, user_map, item_map, df, top_n)
    return {"user_id": user_id, "recommendations": result}
