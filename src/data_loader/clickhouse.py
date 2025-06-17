from clickhouse_driver import Client
import pandas as pd

def load_clickhouse_events():
    client = Client('clickhouse')
    query = """
    SELECT 
      anon_id AS user_id,
      tracking_key,
      tracking_type,
      common_timestamp
    FROM tracking.trackings
    WHERE tracking_type IN ('view', 'cart', 'wish', 'purchase')
      AND common_timestamp >= now() - INTERVAL 30 DAY
    """
    result = client.execute(query)
    df = pd.DataFrame(result, columns=['user_id', 'tracking_key', 'tracking_key', 'common_timestamp'])
    return df