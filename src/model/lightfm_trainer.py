from lightfm import LightFM

def train_model(interaction_matrix):
    model = LightFM(loss='warp')
    model.fit(interaction_matrix, epochs=10, num_threads=4)
    return model