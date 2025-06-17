import numpy as np

def recommend_fallback(df, top_n=10):
    weights = {'view': 1, 'cart': 3, 'purchase': 5}
    df['weight'] = df['tracking_type'].map(weights)

    top_items = (
        df.groupby('tracking_key')['weight']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    return top_items

# ✅ 사용자 추천 (Cold Start 대응 포함)
def recommend(model, user_id, interaction_matrix, user_map, item_map, df, top_n=10):
    reverse_item_map = {v: k for k, v in item_map.items()}
    user_idx = user_map.get(user_id)

    if user_idx is None:
        print(f"🔁 Cold start: user '{user_id}' not found. Returning popular items.")
        return recommend_fallback(df, top_n)

    scores = model.predict(user_idx, np.arange(interaction_matrix.shape[1]))
    top_items = np.argsort(-scores)[:top_n]
    return [reverse_item_map[i] for i in top_items]
