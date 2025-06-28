import os
import pickle

from app.config import settings
from core.preprocess.transformer import transform_interaction_matrix
from app.utils.model_utils import find_latest_version_dir
from app.utils.clickhouse import load_clickhouse_events
from core.model.recommender import recommend as core_recommend

def get_recommendations(site_id: str, top_k: int):
    # 최신 버전 디렉터리 찾기 (find_latest_version_dir도 model_utils 에 정의했다고 가정)
    latest_dir = find_latest_version_dir(site_id, base_dir=settings.MODEL_BASE_DIR)
    version = os.path.basename(latest_dir)

    # 모델 파일 로드 헬퍼
    def load(p):
        with open(p, "rb") as f:
            return pickle.load(f)

    model    = load(os.path.join(latest_dir, "model.pkl"))
    user_map = load(os.path.join(latest_dir, "user_map.pkl"))
    item_map = load(os.path.join(latest_dir, "item_map.pkl"))

    # ClickHouse 이벤트 불러와서 행렬 변환
    df = load_clickhouse_events(site_filter=site_id)

    if df.empty:
        return {
            "site_id": site_id,
            "version": version,
            "recommended_items": []
        }

    matrix, _, _ = transform_interaction_matrix(df)

    # 추천 로직 호출
    items = core_recommend(
        model, site_id, matrix, user_map, item_map, df, top_k=top_k
    )
    return {
        "site_id": site_id,
        "version": version,
        "recommended_items": items
    }
