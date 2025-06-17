from datetime import datetime, timedelta
import random
import uuid
import pandas as pd

def generate_mock_tracking_data(n_users=100, n_items=50, n_rows=1000):
    events = ['view', 'cart', 'purchase']
    weights = [0.6, 0.3, 0.1]
    data = []
    now = datetime.utcnow()

    user_ids = [str(uuid.uuid4()) for _ in range(n_users)]
    item_ids = [f"item_{i + 1}" for i in range(n_items)]

    for _ in range(n_rows):
        user_id = random.choice(user_ids)
        item_id = random.choice(item_ids)
        tracking_type = random.choices(events, weights)[0]
        event_time = now - timedelta(days=random.randint(0, 30), seconds=random.randint(0, 86400))
        data.append({
            "user_id": user_id,
            "tracking_key": item_id,
            "tracking_type": tracking_type,
            "common_timestamp": event_time
        })
    return pd.DataFrame(data)
