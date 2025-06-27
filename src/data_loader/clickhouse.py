import os
import pandas as pd
from clickhouse_driver import Client

def load_clickhouse_events(site_filter: str = None) -> pd.DataFrame:
    host = os.getenv("CLICKHOUSE_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_PORT", 9000))
    db   = os.getenv("CLICKHOUSE_DB", "default")
    table = os.getenv("CLICKHOUSE_TABLE", "trackings")
    days = int(os.getenv("CLICKHOUSE_DAYS", 30))

    client = Client(host=host, port=port, database=db)

    query = f"""
    SELECT
        anon_id,
        product_code,
        tracking_type
    FROM {db}.{table}
    WHERE common_ts >= now() - INTERVAL {days} DAY
    """

    if site_filter:
        # SQL 인젝션 대비 간단히 escape 처리
        safe_site = site_filter.replace("'", "''")
        query += f" AND site_id = '{safe_site}'"

    try:
        result = client.execute(query)
        return pd.DataFrame(result, columns=['anon_id', 'product_code', 'tracking_type'])
    except Exception as e:
        print("❌ ClickHouse 조회 오류:", e)
        return pd.DataFrame(columns=['anon_id', 'product_code', 'tracking_type'])
