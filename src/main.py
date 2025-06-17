from src.data_loader.clickhouse import load_clickhouse_events
from src.preprocess.transformer import transform_interaction_matrix
from src.model.lightfm_trainer import train_model
from src.model.recommender import recommend
from src.mock.mock_data import generate_mock_tracking_data


if __name__ == '__main__':
    df = generate_mock_tracking_data(n_users=100, n_items=50, n_rows=1000)
    #df = load_clickhouse_events()
    matrix, user_map, item_map = transform_interaction_matrix(df)
    model = train_model(matrix)

    user_id = 123  # 예시용 사용자
    recommendations = recommend(model, user_id, matrix, user_map, item_map, df)
    print(f"추천 상품 for user {user_id}:", recommendations)
