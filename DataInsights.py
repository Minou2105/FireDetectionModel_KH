import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pandas.plotting import table

from utils import *

def get_img_sizes(path_images, path_viz):
    imgs = [img.resolve() for img in Path(path_images).rglob("*") if not img.is_dir()]
    print(len(imgs))
    df_img = pd.DataFrame(data = imgs, columns = ["img_name"])
    df_img["width"] = df_img["img_name"].apply(lambda img_name: Image.open(img_name).width)
    df_img["height"] = df_img["img_name"].apply(lambda img_name: Image.open(img_name).height)

    print(df_img[df_img["width"].isna()])

    #histogram of width and height
    ax_width = df_img["width"].hist(legend = True, bins = 20)
    ax_width.get_figure().savefig(os.path.join(path_viz, "data_insights", "Histogram of images width.jpg"))
    plt.clf()
    ax_height = df_img["height"].hist(legend = True, bins = 20)
    ax_height.get_figure().savefig(os.path.join(path_viz, "data_insights", "Histogram of images height.jpg"))

def get_class_frequencies(train_path, test_path, viz_path):
    path = [("train", train_path),("test", test_path)]
    frequencies = []
    for dataset, dataset_path in path:
        dirs = [(dir.name, dir.resolve()) for dir in Path(dataset_path).iterdir() if dir.is_dir()]
        for dir, dirpath in dirs:
            imgs = [img.resolve() for img in Path(dirpath).glob("*") if not img.is_dir()]
            frequencies.append([dataset, dir, len(imgs)])
    df = pd.DataFrame(data = frequencies, columns=["dataset", "class", "num_samples"])

    #plotting
    ax = plt.subplot(111, frame_on = False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title("Class Frequencies for Train and Test dataset")
    table(ax, df, loc="center")
    plt.savefig(os.path.join(viz_path, "data_insights", "class_frequencies.jpg"))
    return df
