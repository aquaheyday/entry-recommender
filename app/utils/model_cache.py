import os
import pickle
from typing import Tuple
from core.model.lightfm_trainer import load_latest_model
from app.utils.model_utils import load_latest_model_for_lang

_model_store = {}

def load_model_once(site_id: str, lang: str = "und", base_dir: str = "/app/models"
                   ) -> Tuple:  # (model, user_map, item_map)
    key = f"{site_id}:{lang}"
    if key in _model_store:
        return _model_store[key]

    # 최신 버전 디렉터리 찾기
    version_dir = load_latest_model_for_lang(site_id=site_id, lang=lang, base_dir=base_dir)

    # load_latest_model 은 (model, user_map, item_map) 을 반환
    model, user_map, item_map = load_latest_model(version_dir)
    _model_store[key] = (model, user_map, item_map)
    return model, user_map, item_map
