import os
from ..config import settings

def find_latest_version_dir(site_id: str, base_dir: str = None) -> str:
    """
    프로젝트 루트의 'models' 디렉터리 아래에 있는 'site-{site_id}/v*' 중 가장 최신 버전을 반환합니다.
    settings.MODEL_BASE_DIR이 설정되어 있으면 해당 경로를, 아니면 프로젝트 루트의 'models'를 기본으로 사용합니다.
    """
    # 기본 모델 베이스 디렉터리 결정
    if base_dir:
        base = base_dir
    elif hasattr(settings, 'MODEL_BASE_DIR') and settings.MODEL_BASE_DIR:
        base = settings.MODEL_BASE_DIR
    else:
        # config.py 위치 기준으로 프로젝트 루트를 찾고 'models' 디렉터리 지정
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        base = os.path.join(project_root, 'models')

    base = os.path.abspath(base)
    site_dir = os.path.join(base, f"site-{site_id}")
    if not os.path.isdir(site_dir):
        raise FileNotFoundError(f"{site_dir} not found")

    # 버전 디렉터리 목록 수집
    versions = [int(d[1:]) for d in os.listdir(site_dir)
                if d.startswith('v') and d[1:].isdigit()]
    if not versions:
        raise FileNotFoundError(f"No version dirs under {site_dir}")

    latest_version = max(versions)
    return os.path.join(site_dir, f"v{latest_version}")


def safe_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"
