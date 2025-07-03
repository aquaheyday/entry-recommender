import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


def transform_interaction_matrix(df: pd.DataFrame):
    """
    DataFrame에서 anon_id와 product_code 컬럼을 바탕으로
    희소 행렬과 매핑 정보를 반환합니다.
    빈 DataFrame인 경우에도 빈 행렬과 빈 매핑(dict)을 반환합니다.
    """
    # 디버깅 프린트: 함수 호출 여부와 df 상태 확인
    print(f"[DEBUG] transform_interaction_matrix called: df is None={df is None}, empty={{False if df is None else df.empty}}")

    # None 또는 빈 DataFrame 처리: 항상 (matrix, user_map, item_map) 튜플 반환
    if df is None or df.empty:
        return csr_matrix((0, 0)), {}, {}

    # 고유 사용자·아이템 매핑 생성
    users = df['anon_id'].unique().tolist()
    items = df['product_code'].unique().tolist()
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx for idx, i in enumerate(items)}

    # 행렬 인덱스 및 데이터 생성
    row_idx = df['anon_id'].map(user_map)
    col_idx = df['product_code'].map(item_map)
    # 데이터 개수 계산: 이벤트 수
    event_count = df.shape[0]
    data = [1] * event_count

    # 행렬 크기 계산: 고유 사용자·아이템 수
    n_users = df['anon_id'].nunique()
    n_items = df['product_code'].nunique()

    # COO -> CSR 변환
    matrix = coo_matrix((data, (row_idx, col_idx)), shape=(n_users, n_items)).tocsr()
    return matrix, user_map, item_map
