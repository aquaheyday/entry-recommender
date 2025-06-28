import pandas as pd
from scipy.sparse import coo_matrix

def transform_interaction_matrix(df: pd.DataFrame):
    users = df['anon_id'].unique().tolist()
    items = df['product_code'].unique().tolist()
    user_map = {u:i for i,u in enumerate(users)}
    item_map = {i:j for j,i in enumerate(items)}

    rows = df['anon_id'].map(user_map)
    cols = df['product_code'].map(item_map)
    data = [1] * len(df)

    matrix = coo_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    return matrix, user_map, item_map
