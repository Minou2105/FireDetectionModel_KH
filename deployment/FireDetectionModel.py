import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, Input, Rescaling, GlobalAveragePooling2D
import keras
from keras.applications.resnet import ResNet50
import numpy as np


class FireDetectionModel(tf.keras.Model):
    def __init__(self, input_shape, use_resnet=False):
        super().__init__()
        self.custom_input_shape = input_shape
        self.use_resnet = use_resnet
        self.custom_model = self.create_model()

    def create_model(self):
        if self.use_resnet:
            base_model = ResNet50(weights="imagenet",
                                  include_top=False,
                                  input_shape=self.custom_input_shape)
            base_model.trainable = False

            inputs = Input(shape=self.custom_input_shape)
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            outputs = Dense(3, activation="softmax")(x)
            model = keras.Model(inputs, outputs)

        else:
            model = tf.keras.models.Sequential([
                Rescaling(1 / 255, input_shape=self.custom_input_shape),
                Conv2D(64, (3, 3), activation="relu", name="conv1", padding="same"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.4),
                Conv2D(128, (3, 3), activation="relu", name="conv2", padding="same"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.5),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.4),
                Dense(3, activation="softmax")
            ])
        return model

    def decode_prediction(self, prediction):
        index = np.argmax(prediction)
        classes = ["fire","Smoke", "no fire"]
        return classes[index]




    def call(self, x):
        return self.custom_model(x)
