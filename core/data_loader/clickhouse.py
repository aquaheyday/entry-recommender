import pandas as pd
from clickhouse_driver import Client

def load_clickhouse_events(site_filter: str = None) -> pd.DataFrame:
    client = Client(host='clickhouse', port=9000, database='tracking')
    query = """
    SELECT anon_id, product_code, tracking_type
      FROM tracking.trackings
     WHERE common_ts >= now() - INTERVAL 30 DAY
    """
    if site_filter:
        sf = site_filter.replace("'", "''")
        query += f" AND site_id = '{sf}'"
    rows = client.execute(query)
    return pd.DataFrame(rows, columns=['anon_id','product_code','tracking_type'])
