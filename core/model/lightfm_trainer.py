from lightfm import LightFM

def train_model(matrix):
    model = LightFM(no_components=30, learning_rate=0.05, loss='warp')
    model.fit(matrix, epochs=10, num_threads=4)
    return model
