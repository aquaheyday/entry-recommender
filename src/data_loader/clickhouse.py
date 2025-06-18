from clickhouse_driver import Client
import pandas as pd

def load_clickhouse_events(site_filter: str = None) -> pd.DataFrame:
    client = Client(host='clickhouse')  # 도커 네트워크 이름에 맞게 설정

    # 기본 쿼리
    query = """
    SELECT
        anon_id,
        product_code,
        tracking_type
    FROM tracking.trackings
    WHERE common_ts >= now() - INTERVAL 30 DAY
    """

    if site_filter:
        query += f" AND site_id = '{site_filter}'"

    try:
        result = client.execute(query)
        df = pd.DataFrame(result, columns=['anon_id', 'product_code', 'tracking_type'])
        return df
    except Exception as e:
        print("❌ ClickHouse 조회 오류:", e)
        return pd.DataFrame(columns=['anon_id', 'product_code', 'tracking_type'])
