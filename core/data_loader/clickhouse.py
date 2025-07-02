import os
import pandas as pd
from clickhouse_driver import Client

def load_popular_items(
    tracking_filter: str,
    top_k: int = 10,
    days: int = 30
) -> pd.DataFrame:
    """
    tracking_key 별로 최근 days일간 product_code별 카운트를 집계,
    상위 top_k개의 product_code와 카운트를 DataFrame으로 반환.
    """
    client = Client(
        host='clickhouse',
        port=9000,
        database=os.getenv("CLICKHOUSE_DB", "tracking"),
    )
    # SQL 인젝션 방어를 위해 작은따옴표를 이스케이프
    sf = tracking_filter.replace("'", "''")
    sql = f"""
    SELECT
      product_code,
      count() AS cnt
    FROM tracking.trackings
    WHERE common_ts >= now() - INTERVAL {days} DAY
    AND tracking_key = '{sf}'
    AND product_code IS NOT NULL
    AND product_code != ''
    GROUP BY product_code
    ORDER BY cnt DESC
    LIMIT {top_k}
    """
    try:
        rows = client.execute(sql)
        return pd.DataFrame(rows, columns=["product_code", "cnt"])
    except Exception as e:
        # 로거가 있으면 logger.error로 바꿔주세요
        print(f"❌ load_popular_items 오류: {e}")
        return pd.DataFrame(columns=["product_code", "cnt"])

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
      common_page_language,
      common_site_domain,
      common_protocol
    FROM tracking.trackings
    WHERE common_ts >= now() - INTERVAL 30 DAY
    AND product_code IS NOT NULL
    AND product_code != ''
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
            'common_page_language',
            'common_site_domain',
            'common_protocol'
        ]
    )

def load_clickhouse_item_metadata(tracking_filter: str) -> pd.DataFrame:
    """
    tracking_key 별로 distinct product_code 에 대해
    product_name, price, sold_out, 이미지, 브랜드,
    category_1/2/3 코드·이름, tracking_type, common_page_language
    를 모두 리턴합니다.
    """
    client = Client(
        host='clickhouse',
        port=9000,
        database=os.getenv("CLICKHOUSE_DB", "tracking"),
    )
    sf = tracking_filter.replace("'", "''")
    sql = f"""
    SELECT
      product_code                                 AS product_code,
      anyHeavy(product_name)                       AS product_name,
      anyHeavy(product_price)                      AS product_price,
      anyHeavy(product_dc_price)                   AS product_dc_price,
      anyHeavy(product_sold_out)                   AS product_sold_out,
      anyHeavy(product_image_url)                  AS product_image_url,
      anyHeavy(product_brand)                      AS product_brand,
      anyHeavy(product_category_1_code)            AS product_category_1_code,
      anyHeavy(product_category_1_name)            AS product_category_1_name,
      anyHeavy(product_category_2_code)            AS product_category_2_code,
      anyHeavy(product_category_2_name)            AS product_category_2_name,
      anyHeavy(product_category_3_code)            AS product_category_3_code,
      anyHeavy(product_category_3_name)            AS product_category_3_name,
    FROM tracking.trackings
    WHERE tracking_key = '{sf}'
    AND product_code IS NOT NULL
    AND product_code != ''
    GROUP BY product_code
    """
    try:
        rows = client.execute(sql)
        return pd.DataFrame(rows, columns=[
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
        ])
    except Exception as e:
        # logger.error로 바꿔주세요
        print(f"❌ load_clickhouse_item_metadata 오류: {e}")
        return pd.DataFrame(columns=[
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
        ])

def load_item_metadata_full(
    tracking_key: str,
    lang: str | None = None
) -> pd.DataFrame:
    client = Client(host='clickhouse', port=9000, database='tracking')
    sf = tracking_key.replace("'", "''")

    sql = f"""
    SELECT
        product_code                                AS product_code,
        anyHeavy(product_name)                      AS product_name,
        anyHeavy(product_price)                     AS product_price,
        anyHeavy(product_dc_price)                  AS product_dc_price,
        anyHeavy(product_sold_out)                  AS product_sold_out,
        anyHeavy(product_image_url)                 AS product_image_url,
        anyHeavy(product_brand)                     AS product_brand,
        anyHeavy(product_category_1_code)           AS product_category_1_code,
        anyHeavy(product_category_1_name)           AS product_category_1_name,
        anyHeavy(product_category_2_code)           AS product_category_2_code,
        anyHeavy(product_category_2_name)           AS product_category_2_name,
        anyHeavy(product_category_3_code)           AS product_category_3_code,
        anyHeavy(product_category_3_name)           AS product_category_3_name,
        anyHeavy(tracking_type)                     AS tracking_type
        anyHeavy(common_site_domain)                AS site_domain,
        anyHeavy(common_protocol)                AS protocol,
    FROM tracking.trackings
    WHERE tracking_key = '{sf}'
    AND product_code IS NOT NULL
    AND product_code != ''
    AND common_ts >= now() - INTERVAL 30 DAY
    """

    # raw 컬럼(common_page_language) 로 필터를 걸어 줍니다
    if lang:
        lang_escaped = lang.replace("'", "''")
        sql += f"\n  AND common_page_language = '{lang_escaped}'"

    sql += "\nGROUP BY product_code"

    rows = client.execute(sql)
    return pd.DataFrame(
        rows,
        columns=[
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
            'site_domain',
            'protocol',
        ]
    )
