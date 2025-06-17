# train_recommendation_model.py
from src.data_loader.clickhouse import load_clickhouse_events
from src.preprocess.transformer import transform_interaction_matrix
from src.model.lightfm_trainer import train_model
from src.utils.model_utils import save_model_and_mappings

df = load_clickhouse_events()
matrix, user_map, item_map = transform_interaction_matrix(df)
model = train_model(matrix)
save_model_and_mappings(model, user_map, item_map)
print("✅ 모델 학습 및 저장 완료")
