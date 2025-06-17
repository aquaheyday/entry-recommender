import pickle
import os

def save_model_and_mappings(model, user_map, item_map):
    os.makedirs("models", exist_ok=True)
    with open("models/lightfm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open("models/item_map.pkl", "wb") as f:
        pickle.dump(item_map, f)

def load_model_and_mappings():
    with open("models/lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/user_map.pkl", "rb") as f:
        user_map = pickle.load(f)
    with open("models/item_map.pkl", "rb") as f:
        item_map = pickle.load(f)
    return model, user_map, item_map
