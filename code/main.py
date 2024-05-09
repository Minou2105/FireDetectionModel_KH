import os

from tensorflow.python.client import device_lib
import tensorflow as tf

from DataInsights import get_img_sizes
from FireDetectionModel import FireDetectionModel
from Preprocessing import read_images_from_dir, create_all_dirs
from utils import read_config, output_to_file, output_as_before


def get_available_devices():
    print(tf.config.list_physical_devices("GPU"))
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

def test():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

def test_fire_initial():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


    config = read_config()
    create_all_dirs(config)
    image_input_path = config["input"]["image_input_path"]
    viz_output_path = config["output"]["viz_path"]
    logging_path = config["output"]["logging_path"]
    image_size = config["input"]["image_size"]
    batch_size = config["input"]["batch_size"]

    filename = "Initial_Run.txt"
    std_orig, file = output_to_file(os.path.join(logging_path, filename))

    get_img_sizes(image_input_path, viz_output_path)

    train_ds = read_images_from_dir(os.path.join(image_input_path, "train"), image_size = image_size, batch_size = batch_size)
    test_ds = read_images_from_dir(os.path.join(image_input_path, "test"), image_size = image_size, batch_size = batch_size)

    for batch_images, batch_labels in train_ds.take(1):  # Take one batch for inspection
        print("Batch images shape:", batch_images.shape)  # Should print (32, 256, 256, 3) if batch size is 32
        print("Batch labels shape:", batch_labels.shape)

    #train_ds = train_ds.batch(32)
    #test_ds = test_ds.batch(32)

    model = FireDetectionModel(input_shape = (image_size[0], image_size[1], 3))
    model.build(input_shape = (None,image_size[0], image_size[1], 3))
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    hist = model.fit(train_ds, epochs = 4)
    output_as_before(std_orig, file)

if __name__ == "__main__":
    print(get_available_devices())

    test_fire_initial()








