import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def recommend(df: pd.DataFrame, top_k: int = 10):
    """
    사이트 전체 이벤트 DataFrame으로부터 인기 상품 top_k 리스트를 반환합니다.

    Args:
        df (pd.DataFrame): 'product_code' 컬럼을 포함하는 이벤트 데이터
        top_k (int): 상위 몇 개 상품을 반환할지

    Returns:
        List[str]: 상위 top_k 상품 코드 리스트
    """
    # pandas alias 보장
    # df가 DataFrame인지 확인
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.info("No data available for recommendation, returning empty list")
        return []

    # 상품별 등장 횟수 집계 및 상위 top_k 추출
    counts = df['product_code'].value_counts()
    top_items = counts.head(top_k).index.tolist()
    logger.info("Computed top %d popular items: %s", top_k, top_items)

    return top_items