import argparse
import itertools

import torch
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ruamel.yaml import YAML
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from skimage.transform import rescale

interp_mode = transforms.InterpolationMode


GA_ROOT = "../data/"
PREPROCESS_ROOT = GA_ROOT + "preprocess/"
SPLITS_ROOT = GA_ROOT + "splits/"


classes = [
    "bg",
    "outer",
    "inner",
    "window",
    "door",
    "frame",
    "room",
    "symbol",
    "unknown",
]

gray_cmap = plt.get_cmap("gray")
nipy_cmap = plt.get_cmap("nipy_spectral")
random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--floorplan_id", dest="floorplan_id", action="store", type=str)
    parser.add_argument("--method", dest="method", action="store", type=str)
    parser.add_argument("--split_id", dest="split_id", action="store", type=int)
    parser.add_argument("--hparam_f", dest="hparam_f", action="store", type=str)
    parser.add_argument("--gan_f", dest="gan_f", action="store", type=str)
    parser.add_argument("--topo_f", dest="topo_f", action="store", type=str)
    parser.add_argument("--debug", dest="debug", action="store_true", default=False)
    parser.add_argument("--resume", dest="resume", action="store_true", default=False)
    parser.add_argument("--restart", dest="restart", action="store_true", default=False)
    parser.add_argument("--full", dest="full", action="store_true", default=False)
    parser.add_argument(
        "--save_full", dest="save_full", action="store_true", default=False
    )

    args = parser.parse_args()

    return args


def parse_config(config_path):
    yaml = YAML(typ="safe")
    config = yaml.load(open(config_path, "r"))

    return config


def dict_to_int_keys(dictionary):
    new_dictionary = {}

    for key, value in dictionary.items():
        new_dictionary[int(key)] = value

    return new_dictionary


# by default it performs NN resizing
def resize(image, size, interpolation=interp_mode.NEAREST):
    if len(image.shape) == 2:
        unsqueezed = True
        image = torch.tensor(image).unsqueeze(0)
    elif len(image.shape) == 3:
        unsqueezed = False
        image = torch.tensor(image).permute([2, 0, 1])
    else:
        raise Exception("Unknown image format")

    resizer = transforms.Resize(size, interpolation=interpolation)
    image = resizer(image)

    if unsqueezed:
        return image.squeeze().numpy()
    else:
        return image.permute([1, 2, 0]).numpy()


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


def vis_input(sample):
    given_masks = sample["given_masks"]
    ind_masks = sample["ind_masks"]
    gen_masks = sample["gen_masks"]
    ignore_masks = sample["ignore_masks"]
    given_imgs = sample["given_imgs"]
    full_masks = sample["full_masks"]
    semantics = sample["semantics"]

    ind_masks = ind_masks.unsqueeze(1).unsqueeze(1)
    ind_masks = ind_masks.repeat([1, 64, 64])

    rows = []
    h, w = full_masks[0].shape

    semantic_given = np.zeros([h, w], dtype=int)
    semantic_gen = np.zeros([h, w], dtype=int)
    semantic_full = np.zeros([h, w], dtype=int)

    for given_mask, ind_mask, ignore_mask, gen_mask, full_mask, sem in zip(
        given_masks, ind_masks, ignore_masks, gen_masks, full_masks, semantics
    ):
        given_mask = given_mask.cpu().numpy()
        ind_mask = ind_mask.cpu().numpy()
        ignore_mask = ignore_mask.cpu().numpy()
        gen_mask = gen_mask.cpu().numpy()
        full_mask = full_mask.cpu().numpy()
        sem = sem.argmax().cpu().numpy()

        semantic_given[given_mask > 0] = sem
        semantic_gen[gen_mask > 0] = sem
        semantic_full[full_mask > 0] = sem
        semantic_given[given_mask == 0] = 8

        given_color = (given_mask + 1) / 2
        assert (given_color >= 0).all()
        given_color = gray_cmap(given_color)[:, :, :3] * 255.0
        given_color = given_color.astype(np.uint8)

        ind_color = gray_cmap(ind_mask)[:, :, :3] * 255.0
        ind_color = ind_color.astype(np.uint8)

        # given_img = (given_img + 1) / 2
        # assert (given_img >= 0).all()
        # given_img = gray_cmap(given_img)[:, :, :3] * 255.
        # given_img = given_img.astype(np.uint8)

        gen_color = (gen_mask + 1) / 2
        assert (gen_color >= 0).all()
        gen_color = gray_cmap(gen_color)[:, :, :3] * 255.0
        gen_color = gen_color.astype(np.uint8)

        full_color = (full_mask + 1) / 2
        assert (full_color >= 0).all()
        full_color = gray_cmap(full_color)[:, :, :3] * 255.0
        full_color = full_color.astype(np.uint8)

        l2_valid = full_color.copy()
        l2_valid[ignore_mask] = [255, 0, 0]

        given_color = np.pad(given_color, [[2, 2], [2, 2], [0, 0]], constant_values=127)
        ind_color = np.pad(ind_color, [[2, 2], [2, 2], [0, 0]], constant_values=127)
        # given_img = np.pad(given_img, [[2,2], [2,2], [0,0]], constant_values=127)
        gen_color = np.pad(gen_color, [[2, 2], [2, 2], [0, 0]], constant_values=127)
        full_color = np.pad(full_color, [[2, 2], [2, 2], [0, 0]], constant_values=127)
        l2_valid = np.pad(l2_valid, [[2, 2], [2, 2], [0, 0]], constant_values=127)

        row = np.concatenate(
            [given_color, ind_color, gen_color, full_color, l2_valid], axis=1
        )
        row = rescale(row, [4, 4, 1], order=0, preserve_range=True, anti_aliasing=False)
        rows.append(row)

    l2_valid_all = semantic_full.copy()
    l2_ignore_mask = ignore_masks.sum(0).numpy().astype(bool)
    l2_valid_all[l2_ignore_mask] = 8

    image_given = nipy_cmap(semantic_given / 8.0)[:, :, :3] * 255.0
    image_gen = nipy_cmap(semantic_gen / 8.0)[:, :, :3] * 255.0
    image_full = nipy_cmap(semantic_full / 8.0)[:, :, :3] * 255.0
    image_ignore = nipy_cmap(l2_valid_all / 8.0)[:, :, :3] * 255.0
    blank = np.zeros_like(image_full)

    image_given = np.pad(image_given, [[2, 2], [2, 2], [0, 0]], constant_values=127)
    image_gen = np.pad(image_gen, [[2, 2], [2, 2], [0, 0]], constant_values=127)
    image_full = np.pad(image_full, [[2, 2], [2, 2], [0, 0]], constant_values=127)
    image_ignore = np.pad(image_ignore, [[2, 2], [2, 2], [0, 0]], constant_values=127)
    blank = np.pad(blank, [[2, 2], [2, 2], [0, 0]], constant_values=127)

    row = np.concatenate(
        [image_given, image_gen, image_full, image_ignore, blank], axis=1
    )
    row = rescale(row, [4, 4, 1], order=0, preserve_range=True, anti_aliasing=False)
    rows.append(row)

    # save full vis
    full_vis = np.concatenate(rows, axis=0)
    # full_vis = Image.fromarray(full_vis.astype('uint8'))

    return full_vis.astype("uint8")


def save_gif(generator_inputs, step_vis, fname):
    h, w = generator_inputs["full_masks"][0].shape
    semantic_given = np.zeros([h, w], dtype=int)
    semantic_full = np.zeros([h, w], dtype=int)

    given_masks = generator_inputs["given_masks"]
    full_masks = generator_inputs["full_masks"]
    semantics = generator_inputs["semantics"]
    # gt_mask = generator_inputs['gt_mask'].cpu().numpy()

    for given_mask, full_mask, semantic in zip(given_masks, full_masks, semantics):
        given_mask = given_mask.cpu().numpy()
        full_mask = full_mask.cpu().numpy()
        semantic = semantic.argmax().cpu().numpy()

        semantic_given[given_mask > 0] = semantic
        semantic_full[full_mask > 0] = semantic

    image_given = nipy_cmap(semantic_given / 8.0)[:, :, :3] * 255.0
    image_given = np.pad(image_given, [[2, 2], [2, 2], [0, 0]])
    image_given = rescale(
        image_given, [4, 4, 1], order=0, preserve_range=True, anti_aliasing=False
    )
    image_given = image_given.astype("uint8")

    image_full = nipy_cmap(semantic_full / 8.0)[:, :, :3] * 255.0
    image_full = np.pad(image_full, [[2, 2], [2, 2], [0, 0]])
    image_full = rescale(
        image_full, [4, 4, 1], order=0, preserve_range=True, anti_aliasing=False
    )
    image_full = image_full.astype("uint8")

    image_gt = nipy_cmap(semantic_full / 8.0)[:, :, :3] * 255.0
    image_gt = np.pad(image_gt, [[2, 2], [2, 2], [0, 0]])
    image_gt = rescale(
        image_gt, [4, 4, 1], order=0, preserve_range=True, anti_aliasing=False
    )
    image_gt = image_gt.astype("uint8")

    frames = []

    for semantic_frame in step_vis:
        # semantic_frame[semantic_frame < 0] = 0
        semantic_frame += 1
        image_frame = nipy_cmap(semantic_frame / 9.0)[:, :, :3] * 255.0
        image_frame = np.pad(image_frame, [[2, 2], [2, 2], [0, 0]])
        image_frame = rescale(
            image_frame, [4, 4, 1], order=0, preserve_range=True, anti_aliasing=False
        )
        image_frame = image_frame.astype("uint8")

        frame = np.concatenate([image_given, image_frame, image_full, image_gt], axis=1)
        frames.append(frame)

    frames.insert(0, np.zeros_like(frames[0]))
    imageio.mimsave(fname, frames, duration=0.2)
