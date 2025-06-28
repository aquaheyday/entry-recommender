import os
import pandas as pd
from clickhouse_driver import Client
from .model_utils import safe_quote

from ..config import settings


def load_clickhouse_events(site_filter: str = None) -> pd.DataFrame:
    client = Client(
        host=settings.CLICKHOUSE_HOST,
        port=settings.CLICKHOUSE_PORT,
        database=os.getenv("CLICKHOUSE_DB", "tracking"),
    )
    query = f"""
    SELECT anon_id, product_code, tracking_type
      FROM tracking.trackings
     WHERE common_ts >= now() - INTERVAL 30 DAY
    """
    if site_filter:
        query += f" AND site_id = {safe_quote(site_filter)}"
    try:
        data = client.execute(query)
        return pd.DataFrame(data, columns=['anon_id','product_code','tracking_type'])
    except Exception as e:
        print("❌ ClickHouse error:", e)
        return pd.DataFrame(columns=['anon_id','product_code','tracking_type'])
