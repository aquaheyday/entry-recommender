from scipy.sparse import coo_matrix

def transform_interaction_matrix(df):
    if df.empty:
        return coo_matrix((0, 0)), {}, {}

    user_map = {id_: idx for idx, id_ in enumerate(df['user_id'].unique())}
    item_map = {id_: idx for idx, id_ in enumerate(df['tracking_key'].unique())}

    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['tracking_key'].map(item_map)

    df = df.dropna(subset=['user_idx', 'item_idx'])

    if df.empty or df['user_idx'].isna().all() or df['item_idx'].isna().all():
        return coo_matrix((0, 0)), user_map, item_map

    row = df['user_idx'].astype(int).values
    col = df['item_idx'].astype(int).values
    data = [1] * len(df)

    if len(row) == 0 or len(col) == 0:
        return coo_matrix((0, 0)), user_map, item_map

    matrix = coo_matrix((data, (row, col)))

    return matrix, user_map, item_map
