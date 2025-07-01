import os
from ..config import settings


def find_latest_version_dir(site_id: str, lang: str = 'und', base_dir: str = None) -> str:
    """
    모델 저장소(base_dir) 아래 '{site_id}/<lang>/v*' 중 가장 최신 버전을 반환합니다.
    settings.MODEL_BASE_DIR 이 설정되어 있으면 해당 경로를 사용하고,
    아니면 프로젝트 루트의 'models' 디렉터리를 기본으로 사용합니다.
    """
    # 기본 모델 베이스 디렉터리 결정
    if base_dir:
        base = base_dir
    elif getattr(settings, 'MODEL_BASE_DIR', None):
        base = settings.MODEL_BASE_DIR
    else:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        base = os.path.join(project_root, 'models')

    base = os.path.abspath(base)
    site_lang_dir = os.path.join(base, site_id, lang)
    if not os.path.isdir(site_lang_dir):
        # 언어별 디렉토리가 없으면 und 로 폴백
        fallback = os.path.join(base, site_id, 'und')
        if not os.path.isdir(fallback):
            raise FileNotFoundError(
                f"Neither {site_lang_dir} nor fallback {fallback} exist"
            )
        site_lang_dir = fallback

    # 버전 디렉터리 목록 수집
    version_dirs = [d for d in os.listdir(site_lang_dir)
                    if d.startswith('v') and d[1:].isdigit()]
    if not version_dirs:
        raise FileNotFoundError(f"No version dirs under {site_lang_dir}")

    versions = sorted(int(d[1:]) for d in version_dirs)
    latest = versions[-1]
    return os.path.join(site_lang_dir, f"v{latest}")


def safe_quote(s: str) -> str:
    """SQL 쿼리에서 안전한 문자열 인용을 위해 작은따옴표 이스케이프"""
    return "'" + s.replace("'", "''") + "'"
