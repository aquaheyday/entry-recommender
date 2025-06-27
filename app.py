from fastapi import FastAPI, HTTPException
from src.utils.model_utils import load_model_and_mappings
from src.data_loader.clickhouse import load_clickhouse_events
from src.preprocess.transformer import transform_interaction_matrix
from src.train_user import train_model_for_site
from pydantic import BaseModel
import pickle
from src.model.recommender import recommend


import os

app = FastAPI()

# 요청 바디 구조 먼저 정의해야 함!
class RecommendationRequest(BaseModel):
    site_id: str
    top_k: int = 10

# 최초 모델 및 매핑 로드
model = None
user_map = {}
item_map = {}

@app.post("/train/{site_id}")
def train_user_model(site_id: str):
    try:
        model_path = train_model_for_site(site_id)
        return {
            "message": "✅ 모델 학습 완료",
            "model_path": model_path,
            "site_id": site_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_latest_version_dir(site_id: str, base_dir="models/") -> str:
    """
    models/user_models/site-{site_id} 아래 v숫자 형태 디렉터리 중
    가장 큰 버전을 찾아 절대 경로로 반환.
    """
    site_dir = os.path.join(base_dir, f"site-{site_id}")
    if not os.path.isdir(site_dir):
        raise FileNotFoundError(f"{site_dir} 자체가 없습니다.")
    versions = []
    for name in os.listdir(site_dir):
        if name.startswith("v") and name[1:].isdigit():
            versions.append(int(name[1:]))
    if not versions:
        raise FileNotFoundError(f"{site_dir}에 버전 디렉터리가 하나도 없습니다.")
    latest = max(versions)
    return os.path.join(site_dir, f"v{latest}")

@app.post("/recommend")
def recommend_items(req: RecommendationRequest):
    try:
        latest_dir = find_latest_version_dir(req.site_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 최신 버전 디렉토리 기준으로 경로 설정
    version = os.path.basename(latest_dir)  # e.g. "v3"
    model_path     = os.path.join(latest_dir, f"model_{req.site_id}.pkl")
    user_map_path  = os.path.join(latest_dir, f"user_map_{req.site_id}.pkl")
    item_map_path  = os.path.join(latest_dir, f"item_map_{req.site_id}.pkl")

    # 모델 파일 존재 확인
    for p in (model_path, user_map_path, item_map_path):
        if not os.path.exists(p):
            raise HTTPException(status_code=404,
                                detail=f"❌ {p} 파일이 없습니다.")

    # 파일 로드
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(user_map_path, "rb") as f:
        user_map = pickle.load(f)
    with open(item_map_path, "rb") as f:
        item_map = pickle.load(f)

    # 이벤트 로드
    df = load_clickhouse_events(site_filter=req.site_id)
    if df.empty:
        raise HTTPException(status_code=400,
                            detail=f"❌ 사용자 {req.site_id}에 대한 데이터가 없습니다.")

    # 상호작용 행렬 생성
    matrix, _, _ = transform_interaction_matrix(df)

    # 추천 수행
    try:
        items = recommend(model, req.site_id, matrix, user_map, item_map, df,
                          top_k=req.top_k)
        return {
            "site_id": req.site_id,
            "version": version,
            "recommended_items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"추천 실패: {str(e)}")

@app.on_event("startup")
def startup_event():
    global model, user_map, item_map, df, matrix
    if not os.path.exists("models/lightfm_model.pkl"):
        print("⚠️ 모델이 존재하지 않습니다. 먼저 학습을 수행하세요.")
        return
    model, user_map, item_map = load_model_and_mappings()
    df = load_clickhouse_events()
    matrix, _, _ = transform_interaction_matrix(df)
