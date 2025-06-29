import logging
from app.utils.clickhouse import load_clickhouse_events

logger = logging.getLogger(__name__)

def get_recommendations(site_id: str, top_k: int):
    """
    사이트 전체 이벤트로부터 인기 상품 top_k 리스트를 반환합니다.
    """
    df = load_clickhouse_events(site_filter=site_id)
    if df is None or df.empty:
        logger.info("No events for site %s, returning empty list", site_id)
        return {"site_id": site_id, "recommended_items": []}

    # 상품코드별 빈도 집계
    counts = df["product_code"].value_counts()
    top_items = counts.head(top_k).index.tolist()
    logger.info("Top %d popular items for site %s: %s", top_k, site_id, top_items)

    return {"site_id": site_id, "recommended_items": top_items}
