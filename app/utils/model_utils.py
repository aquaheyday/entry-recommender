import os
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def find_latest_version_dir(
    site_id: str,
    lang: str = 'und',
    base_dir: str = None
) -> str:
    """
    settings.MODEL_BASE_DIR 또는 base_dir(우선)이 가리키는 'lightfm' 폴더 아래에서
    '{site_id}/v*/{lang}' 중 가장 최신 버전을 반환합니다.
    lang 디렉터리가 없으면 'und' 로 폴백합니다.
    """
    # 1) base 경로 결정 (예: /app/models/lightfm)
    if base_dir:
        base = base_dir
    elif getattr(settings, 'MODEL_BASE_DIR', None):
        base = settings.MODEL_BASE_DIR
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        base = os.path.join(project_root, 'models', 'lightfm')
    base = os.path.abspath(base)

    # 2) site root 확인
    site_root = os.path.join(base, site_id)
    logger.info(f"site_root:{site_root}")
    if not os.path.isdir(site_root):
        raise FileNotFoundError(f"No site directory under {base}: {site_id}")

    # 3) version 디렉터리(v*) 목록 수집
    version_dirs = [d for d in os.listdir(site_root)
                    if d.startswith('v') and d[1:].isdigit()]
    if not version_dirs:
        raise FileNotFoundError(f"No version dirs under {site_root}")
    # 가장 큰 버전 선택
    latest_ver = max(int(d[1:]) for d in version_dirs)
    ver_dir = os.path.join(site_root, f"v{latest_ver}")

    # 4) lang 서브디렉터리 확인 (폴백 포함)
    lang_dir = os.path.join(ver_dir, lang)
    if not os.path.isdir(lang_dir):
        raise FileNotFoundError(f"Neither '{os.path.join(ver_dir, lang)}' nor '{lang_dir}' exist")
    logger.info(f"lang_dir:{lang_dir}")
    return lang_dir
