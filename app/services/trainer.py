from ..config import settings
from app.utils.model_utils import find_latest_version_dir
from core.train_user import train_model_for_site as core_train

def train_site_model(site_id: str):
    result = core_train(site_id)
    # result: dict with version, model_path, user_map_path, item_map_path
    return {
        "site_id": site_id,
        **result
    }
