from scipy.sparse import coo_matrix

def transform_interaction_matrix(df):
    if df.empty:
        return coo_matrix((0, 0)), {}, {}

    # ✅ 필수 컬럼 확인
    required_cols = {'anon_id', 'product_code', 'tracking_type'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"❌ Missing required columns in DataFrame: {missing}")

    # ✅ 가중치 설정
    weights = {
        "view": 1,
        "cart": 2,
        "wishlist": 3,
        "conversion": 5,
    }

    # ✅ 가중치 적용
    df['weight'] = df['tracking_type'].map(weights).fillna(1)

    # ✅ 유저 및 아이템 인덱스 매핑
    user_map = {anon_id: idx for idx, anon_id in enumerate(df['anon_id'].unique())}
    item_map = {track_key: idx for idx, track_key in enumerate(df['product_code'].unique())}

    df['user_idx'] = df['anon_id'].map(user_map)
    df['item_idx'] = df['product_code'].map(item_map)

    # ✅ NaN 필터링
    df = df.dropna(subset=['user_idx', 'item_idx', 'weight'])

    if df.empty:
        return coo_matrix((0, 0)), user_map, item_map

    # ✅ 희소 행렬 생성
    row = df['user_idx'].astype(int).values
    col = df['item_idx'].astype(int).values
    data = df['weight'].astype(float).values

    matrix = coo_matrix((data, (row, col)), shape=(len(user_map), len(item_map)))
    return matrix, user_map, item_map
