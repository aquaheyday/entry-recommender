from lightfm import LightFM
import numpy as np
from scipy.sparse import coo_matrix

def recommend_user_items(model: LightFM, user_id: str, df, user_map, item_map, top_k: int = 10):
    # 1. 사용자 인덱스 확인
    if user_id not in user_map:
        print(f"❌ Cold start: 사용자 '{user_id}'의 학습 데이터가 없음.")
        return []

    user_idx = user_map[user_id]

    # 2. 아이템 전체 인덱스
    item_idxs = list(item_map.values())
    item_keys = list(item_map.keys())

    # 3. 예측 점수 계산
    scores = model.predict(user_ids=user_idx, item_ids=item_idxs)

    # 4. 점수 기준 정렬 후 Top-N 선택
    top_indices = np.argsort(-scores)[:top_k]  # 점수 높은 순
    recommended_items = [item_keys[i] for i in top_indices]

    return recommended_items

def recommend_fallback(df, top_k=10):
    weights = {'view': 1, 'cart': 3, 'purchase': 5}
    df['weight'] = df['tracking_type'].map(weights)

    top_items = (
        df.groupby('product_code')['weight']
        .sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index
        .tolist()
    )
    return top_items

# ✅ 사용자 추천 (Cold Start 대응 포함)
def recommend(model, user_id, interaction_matrix, user_map, item_map, df, top_k=10):
    reverse_item_map = {v: k for k, v in item_map.items()}
    user_idx = user_map.get(user_id)

    if user_idx is None:
        print(f"🔁 Cold start: user '{user_id}' not found. Returning popular items.")
        return recommend_fallback(df, top_k)

    scores = model.predict(user_idx, np.arange(interaction_matrix.shape[1]))
    top_items = np.argsort(-scores)[:top_k]
    return [reverse_item_map[i] for i in top_items]
