import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt

from ruamel.yaml import YAML


def parse_config(config_path):
    yaml = YAML(typ="safe")
    config = yaml.load(open(config_path, "r"))

    return config


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        dest="config_path",
        action="store",
        type=str,
        default="./base.yaml",
    )

    parser.add_argument(
        "--floorplan_id", dest="floorplan_id", action="store", type=str, default=""
    )

    parser.add_argument(
        "--test_id", dest="test_id", action="store", type=int, default=0
    )

    parser.add_argument("--restart", dest="restart", action="store_true", default=False)

    parser.add_argument(
        "--test_folder", dest="test_folder", action="store", type=str, default=""
    )

    args = parser.parse_args()

    return args


def get_method_name(config):
    method_name = []

    keys = [
        "model_name",
        "embed_dims",
        "crop_loss",
        "loss_avg",
        "dd",
        "gamma_reg",
        "crop_size",
    ]

    for key, value in config.items():
        if key in keys:
            method_name.append("%s-%s" % (key, str(value)))

    return "_".join(method_name)


# handles 2D array
def to_onehot(indices, num_classes=None):
    if not num_classes:
        num_classes = indices.max() + 1

    # remember the 2D shape and flatten it
    (h, w) = indices.shape
    indices = indices.flatten()

    onehot = np.zeros((indices.size, num_classes), dtype=np.float32)
    onehot[np.arange(indices.size), indices] = 1

    onehot = np.reshape(onehot, [h, w, num_classes])

    return onehot


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return figure
