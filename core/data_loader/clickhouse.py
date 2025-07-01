import pandas as pd
from clickhouse_driver import Client

def load_clickhouse_events(tracking_filter: str = None) -> pd.DataFrame:
    client = Client(host='clickhouse', port=9000, database='tracking')
    query = """
    SELECT 
      anon_id,
      product_code,
      product_name,
      product_price,
      product_dc_price,
      product_sold_out,
      product_image_url,
      product_brand,
      product_category_1_code,
      product_category_1_name,
      product_category_2_code,
      product_category_2_name,
      product_category_3_code,
      product_category_3_name,
      tracking_type,
      common_page_language
    FROM tracking.trackings
    WHERE common_ts >= now() - INTERVAL 30 DAY
    """
    if tracking_filter:
        sf = tracking_filter.replace("'", "''")
        query += f" AND tracking_key = '{sf}'"
    rows = client.execute(query)
    return pd.DataFrame(
        rows,
        columns=[
            'anon_id',
            'product_code',
            'product_name',
            'product_price',
            'product_dc_price',
            'product_sold_out',
            'product_image_url',
            'product_brand',
            'product_category_1_code',
            'product_category_1_name',
            'product_category_2_code',
            'product_category_2_name',
            'product_category_3_code',
            'product_category_3_name',
            'tracking_type',
            'common_page_language'
        ]
    )

def load_clickhouse_item_metadata(tracking_filter: str) -> pd.DataFrame:
    """
    tracking_key 별로 distinct product_code 에 대해
    product_name, price, 이미지, 브랜드와 함께
    category_1, category_2, category_3 을 모두 리턴합니다.
    """
    client = Client(host='clickhouse', port=9000, database='tracking')
    query = f"""
    SELECT
        product_code                     AS product_code,
        anyHeavy(product_name)           AS product_name,
        anyHeavy(product_price)          AS product_price,
        anyHeavy(product_dc_price)       AS product_dc_price,
        anyHeavy(product_image_url)      AS product_image_url,
        anyHeavy(product_brand)          AS product_brand,
        anyHeavy(product_category_1_name) AS product_category_1_name,
        anyHeavy(product_category_2_name) AS product_category_2_name,
        anyHeavy(product_category_3_name) AS product_category_3_name
    FROM tracking.trackings
    WHERE tracking_key = '{tracking_filter}'
      AND product_code IS NOT NULL
    GROUP BY product_code
    """
    rows = client.execute(query)
    return pd.DataFrame(rows, columns=[
        'product_code',
        'product_name',
        'product_price',
        'product_dc_price',
        'product_image_url',
        'product_brand',
        'product_category_1_name',
        'product_category_2_name',
        'product_category_3_name',
    ])