from ..config import settings
from app.utils.model_utils import find_latest_version_dir
from core.train_user import train_models_for_site

def train_site_model(tracking_key: str):
    result = train_models_for_site(tracking_key)

    return {
        "tracking_key": tracking_key,
        **result
    }
