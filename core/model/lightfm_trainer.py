import os
import pickle
from typing import Tuple, Dict, Any
from lightfm import LightFM

def train_model(matrix):
    model = LightFM(no_components=30, learning_rate=0.05, loss='warp')
    model.fit(matrix, epochs=10, num_threads=4)
    return model

def load_latest_model(
    model_dir: str
) -> Tuple[Any, Dict[str, int], Dict[int, str]]:
    """
    model_dir/{site_id} 아래 버전별 디렉터리(v1, v2, ...) 중
    최신 버전의 모델, user_map, item_map을 로드하여 반환합니다.
    """
    # v* 디렉터리 목록 수집
    versions = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("v") and d[1:].isdigit()
    ]
    if not versions:
        raise FileNotFoundError(f"No model versions found in {model_dir}")
    # 가장 큰 숫자 버전(v3 등)을 선택
    latest = sorted(versions, key=lambda x: int(x[1:]))[-1]
    latest_dir = os.path.join(model_dir, latest)

    # 파일 경로
    model_path = os.path.join(latest_dir, "model.pkl")
    user_map_path = os.path.join(latest_dir, "user_map.pkl")
    item_map_path = os.path.join(latest_dir, "item_map.pkl")

    # 파일 존재 확인
    for p in (model_path, user_map_path, item_map_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file {p} not found")

    # 로드
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(user_map_path, "rb") as f:
        user_map = pickle.load(f)  # dict: user_id -> index
    with open(item_map_path, "rb") as f:
        item_map = pickle.load(f)  # dict: item_id -> index

    # item_map은 item_id->idx이므로, idx->item_id로 inverse 맵 생성
    inv_item_map = {idx: item for item, idx in item_map.items()}

    return model, user_map, inv_item_map