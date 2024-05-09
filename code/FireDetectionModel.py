import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, Input


class FireDetectionModel(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.custom_input_shape = input_shape
        self.custom_model = self.create_model()

    def create_model(self):
        model = tf.keras.models.Sequential([
            Conv2D(32, (3,3), activation = "relu", name = "conv1", input_shape = self.custom_input_shape),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation = "relu", name = "conv2"),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation = "relu"),
            Dense(3, activation="softmax")
        ])
        return model

    def call(self, x):
        return self.custom_model(x)
