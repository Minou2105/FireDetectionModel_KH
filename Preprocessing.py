import keras
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

import os
import matplotlib.pyplot as plt
import os
from DataInsights import get_class_frequencies
from utils import *




def read_images_from_dir(path_images, image_size, batch_size):
    ds = image_dataset_from_directory(path_images,
                                          labels = "inferred",
                                          label_mode = "categorical",
                                          class_names = ["fire", "Smoke", "non fire"] ,
                                          batch_size=batch_size,
                                          image_size = image_size) # not supported,  pad_to_aspect_ratio = True)
    AUTOTUNE = tf.data.AUTOTUNE

    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)

    return ds



def visualize_img(ds):
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5, 5))
    for images, labels in ds.take(1):
        for i in range(3):
            for j in range(3):
                ax[i][j].imshow(images[i * 3 + j].numpy().astype("uint8"))
                ax[i][j].set_title(ds.class_names[decrypt_labels(labels[i * 3 + j])])
    plt.show()

def create_all_dirs(config):
    os.makedirs(config["output"]["viz_path"], exist_ok=True)
    os.makedirs(os.path.join(config["output"]["viz_path"], "data_insights"), exist_ok=True)
    os.makedirs(config["output"]["result_path"], exist_ok=True)
    os.makedirs(config["output"]["logging_path"], exist_ok = True)







if __name__ == "__main__":
    config = read_config()
    create_all_dirs(config)
    #image_input_path = config["input"]["image_input_path"]
    #viz_output_path = config["output"]["viz_path"]
    #df_class_frequencies = get_class_frequencies(os.path.join(image_input_path, "train"),
                                           # os.path.join(image_input_path, "test"),
                                           #     viz_output_path )
    #print(df_class_frequencies)
    #ds = read_images_from_dir(os.path.join(config["input"]["image_input_path"], "train"))
    #print(ds.class_names)
    #visualize_img(ds)

    keras.backend.clear_session()


