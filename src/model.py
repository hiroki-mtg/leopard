from tensorflow import keras
from keras import layers


def build_model(
    hidden_layer_sizes=[32, 64],
    kernel_sizes=[(3, 3)],
    activation="relu",
    pool_sizes=[(2, 2)],
    dropout=0.5,
    num_classes=10,
    input_shape=(28, 28, 1)
    ):
  '''
  :return: keras model object
  '''

  model = keras.Sequential(
    [
      keras.Input(shape=input_shape),
      layers.Conv2D(hidden_layer_sizes[0], kernel_size=kernel_sizes[0], activation=activation),
      layers.MaxPooling2D(pool_size=pool_sizes[0]),
      layers.Conv2D(hidden_layer_sizes[-1], kernel_size=kernel_sizes[-1], activation=activation),
      layers.MaxPooling2D(pool_size=pool_sizes[-1]),
      layers.Flatten(),
      layers.Dropout(dropout),
      layers.Dense(num_classes, activation="softmax")
      ]
  )

  return model
  