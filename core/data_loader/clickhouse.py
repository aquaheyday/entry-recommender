import pandas as pd
from clickhouse_driver import Client

def load_clickhouse_events(tracking_filter: str = None) -> pd.DataFrame:
    client = Client(host='clickhouse', port=9000, database='tracking')
    query = """
    SELECT anon_id, product_code, tracking_type, common_page_language
      FROM tracking.trackings
     WHERE common_ts >= now() - INTERVAL 30 DAY
    """
    if tracking_filter:
        sf = tracking_filter.replace("'", "''")
        query += f" AND tracking_key = '{sf}'"
    rows = client.execute(query)
    return pd.DataFrame(rows, columns=['anon_id','product_code','tracking_type','common_page_language'])

def load_clickhouse_item_metadata(tracking_filter: str) -> pd.DataFrame:
    """
    tracking.trackings 테이블에서 tracking_id 기준으로
    distinct product_code별 카테고리 메타데이터를 로드합니다.
    """
    client = Client(host='clickhouse', port=9000, database='tracking')
    # 각 상품의 대표 카테고리를 하나만 뽑기 위해
    # 예: 가장 흔히 등장하는 category_1_name을 선택
    query = f"""
    SELECT
        product_code AS item_id,
        anyHeavy(common_page_language) AS common_page_language,
        anyHeavy(product_category_1_name) AS category_1,
        anyHeavy(product_category_2_name) AS category_2,
        anyHeavy(product_category_3_name) AS category_3
    FROM tracking.trackings
    WHERE tracking_key = '{tracking_filter}'
      AND product_code IS NOT NULL
    GROUP BY product_code
    """
    result = client.execute(query)
    df = pd.DataFrame(
        result,
        columns=["item_id", "common_page_language", "category_1", "category_2", "category_3"]
    )
    # 필요에 따라 category_1만 쓰거나, 다중 카테고리를 합쳐 하나의 컬럼으로 처리해도 됩니다.
    # 예를 들어 category_1을 대표 카테고리로 사용:
    df = df.rename(columns={"category_1": "category"})
    return df[["item_id", "category"]]