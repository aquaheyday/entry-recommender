from ..config import settings
from app.utils.model_utils import find_latest_version_dir
from core.train_user import train_model_for_site as core_train

def train_site_model(site_id: str):
    result = core_train(site_id)

    return {
        "site_id": site_id,
        **result
    }
