import os
import pickle
from typing import Tuple, Dict, Any
from lightfm import LightFM
import logging

logger = logging.getLogger(__name__)

def train_model(matrix):
    model = LightFM(no_components=30, learning_rate=0.05, loss='warp')
    # COO 포맷으로 변환
    interactions = matrix.tocoo()
    sample_weight = matrix.tocoo()

    model.fit(
        interactions=interactions,
        sample_weight=sample_weight,
        epochs=10,
        num_threads=4,
    )
    return model

def load_latest_model(
    model_dir: str
) -> Tuple[Any, Dict[str, int], Dict[int, str], Dict[str, dict]]:
    """
    model_dir 에는 이미 v{n}/{lang} 까지 포함된
    실제 모델 디렉터리가 들어옵니다.
    그 안의 model.pkl, user_map.pkl, item_map.pkl, item_meta.pkl 을 로드하여 반환합니다.

    Returns:
        model: 학습된 LightFM 모델 객체
        user_map: Dict[user_id, user_idx]
        inv_item_map: Dict[item_idx, item_id]
        item_meta: Dict[item_id, {…메타…}]
    """
    # 1) 파일 경로 결정
    model_path     = os.path.join(model_dir, "model.pkl")
    user_map_path  = os.path.join(model_dir, "user_map.pkl")
    item_map_path  = os.path.join(model_dir, "item_map.pkl")
    meta_path      = os.path.join(model_dir, "item_meta.pkl")

    # 2) 파일 존재 확인 (item_meta는 선택적)
    for p in (model_path, user_map_path, item_map_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Expected file not found: {p}")

    # 3) 파일 로드
    with open(model_path,    "rb") as f: model     = pickle.load(f)
    with open(user_map_path, "rb") as f: user_map  = pickle.load(f)
    with open(item_map_path, "rb") as f: item_map  = pickle.load(f)

    # 4) 역맵 생성
    inv_item_map = {
        idx: item_id
        for item_id, idx in item_map.items()
        if item_id
    }

    # 5) item_meta 로드 (없으면 빈 dict)
    if os.path.isfile(meta_path):
        with open(meta_path, "rb") as f:
            item_meta: Dict[str, dict] = pickle.load(f)
    else:
        logger.warning(f"item_meta.pkl not found in {model_dir}, using empty meta")
        item_meta = {}

    return model, user_map, inv_item_map, item_meta