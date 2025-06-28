import numpy as np

def recommend(model, site_id, matrix, user_map, item_map, df, top_k=10):
    uid = user_map.get(site_id)
    scores = model.predict(uid, np.arange(matrix.shape[1]))
    top = np.argsort(-scores)[:top_k]
    # item_map inverse lookup
    inv_items = {v:k for k,v in item_map.items()}
    return [inv_items[i] for i in top]
