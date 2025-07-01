import os
from app.config import settings

def find_latest_version_dir(site_id: str, lang: str = 'und', base_dir: str = None) -> str:
    """
    모델 저장소(base_dir) 아래 '{site_id}/{lang}/v*' 중 가장 최신 버전을 반환합니다.
    settings.MODEL_BASE_DIR 이 설정되어 있으면 해당 경로를 사용하고,
    아니면 프로젝트 루트의 'models' 디렉터리를 기본으로 사용합니다.
    """
    # 1) base 경로 결정
    if base_dir:
        base = base_dir
    elif getattr(settings, 'MODEL_BASE_DIR', None):
        base = settings.MODEL_BASE_DIR
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        base = os.path.join(project_root, 'models')
    base = os.path.abspath(base)

    # 2) 언어별 디렉토리 찾기 (폴백 포함)
    site_lang = os.path.join(base, site_id, lang)
    if not os.path.isdir(site_lang):
        site_lang = os.path.join(base, site_id, 'und')
        if not os.path.isdir(site_lang):
            raise FileNotFoundError(f"Neither '{lang}' nor 'und' dir exists under {os.path.join(base, site_id)}")

    # 3) 버전(v*) 디렉토리 중 가장 큰 값 선택
    dirs = [d for d in os.listdir(site_lang) if d.startswith('v') and d[1:].isdigit()]
    if not dirs:
        raise FileNotFoundError(f"No version dirs under {site_lang}")
    versions = sorted(int(d[1:]) for d in dirs)
    latest = versions[-1]
    return os.path.join(site_lang, f"v{latest}")
