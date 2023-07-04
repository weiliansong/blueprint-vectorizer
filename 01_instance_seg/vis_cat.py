import os
import numpy as np
import torch
from torch.autograd import Variable
from imageio import imsave, imread
from skimage import measure
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skimage.measure
from skimage.transform import rescale
from utils.config import *
from utils.config import Struct, load_config, compose_config_str
from skimage import img_as_ubyte
import cv2
import random
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from bin_mean_shift import Bin_Mean_Shift
import torchvision.transforms as transforms
from skimage.morphology import erosion, dilation, square
import skimage.morphology as sm
from PIL import Image

# from skimage.segmentation import flood, flood_fill
import itertools

import sys

np.set_printoptions(threshold=sys.maxsize)

config_dict = load_config(file_path="utils/config.yaml")
configs = Struct(**config_dict)

if configs.seed:
    torch.manual_seed(configs.seed)
    if configs.use_cuda:
        torch.cuda.manual_seed(configs.seed)
    np.random.seed(configs.seed)
    random.seed(configs.seed)


h, w = 1534, 761  # 0
# h, w = 1494, 1229 # 1
# h, w = 1521, 969 # 2
# h, w = 1776, 937 # 3
# h, w = 1644, 873 # 4
global segmented_index
segmented_index = np.zeros((h, w))


def segment(input, features, name, id):
    """
    This function performs postprocessing (dimensionality reduction and clustering) for a given network
    output. it also visualizes the resulted segmentation along with the original image and the ground truth
    segmentation and saves all the images locally.
    :param input: (3, h, w) ndarray containing rgb data as outputted by the costume datasets
    :param label: (h, w) or (1, h, w) ndarray with the ground truth segmentation
    :param features: (c, h, w) ndarray with the embedded pixels outputted by the network
    :param name: str with the current experiment name
    :param id: an identifier for the current image (for file saving purposes)
    :return: None. all the visualizations are saved locally
    """

    # os.makedirs('visualizations/segmentations/', exist_ok=True)

    d, h, w = features.size()
    # d, h, w = features.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_np = np.asarray(features)
    pred_flat = np.reshape(features_np, [d, -1])
    pred_flat = pred_flat.T

    # Mean-Shift visualization
    flag_cluster_all = True  # True
    meanshift = MeanShift(
        bandwidth=2.0, bin_seeding=True, cluster_all=flag_cluster_all, n_jobs=-1
    )

    clustering = meanshift.fit(pred_flat)
    predict_segmentation = np.reshape(clustering.labels_, [h, w])

    if not flag_cluster_all:
        predict_segmentation[predict_segmentation == -1] = 0

    # pred_ms = pred_ms.astype(np.float32)
    predict_segmentation = predict_segmentation.astype(np.int)
    predict_segmentation += 1

    # ------------------------

    # visualization and evaluation
    image = tensor_to_image(input.cpu()[0])  # input: image

    pred_seg = cv2.resize(
        np.stack(
            [
                colors_256[predict_segmentation, 0],
                colors_256[predict_segmentation, 1],
                colors_256[predict_segmentation, 2],
            ],
            axis=2,
        ),
        (w, h),
    )

    save_pred_seg = predict_segmentation.astype(np.uint8)

    return save_pred_seg


def tensor_to_image(image):
    # image = torch.from_numpy(image) # change
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


Tensor_to_Image = transforms.Compose([transforms.ToPILImage()])


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap


colors_256 = labelcolormap(256)

colors = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [80, 128, 255],
        [255, 230, 180],
        [255, 0, 255],
        [0, 255, 255],
        [100, 0, 0],
        [0, 100, 0],
        [255, 255, 0],
        [50, 150, 0],
        [200, 255, 255],
        [255, 200, 255],
        [128, 128, 80],
        # [0, 50, 128],
        # [0, 100, 100],
        [0, 255, 128],
        [0, 128, 255],
        [255, 0, 128],
        [128, 0, 255],
        [255, 128, 0],
        [128, 255, 0],
        [0, 0, 0],
    ]
)


def merge_tiles(segs, id, pad):
    segs = segs.astype("uint32")

    os.makedirs("visualizations/segmentations_merge/", exist_ok=True)

    h_num = segs.shape[0]
    w_num = segs.shape[1]

    tile_rows = []
    for h_i in range(h_num):
        tile_row = segs[h_i, 0, :, :]

        for w_i in range(1, w_num):
            tile_row_next = segs[h_i, w_i, :, :]
            tile_row = merge_neighbor_row(tile_row, tile_row_next)
            tile_row = process(tile_row, w_i)

        # viz_image(tile_row, h_i)
        tile_rows.append(tile_row)

    tile_col = tile_rows[0]
    for h_i in range(1, h_num):
        # print("\n\n ============================================", h_i, "=============================")
        tile_col_next = tile_rows[h_i]
        tile_col = merge_col(tile_col, tile_col_next)
        tile_col = process(tile_col)

    merge_all = tile_col

    # or merge_all = seq_name_seg(merge_all)
    num_seg = np.unique(merge_all)
    num_seg = np.asarray(num_seg)

    for i in range(num_seg.shape[0]):
        merge_all[merge_all == num_seg[i]] = i

    merge_all = merge_all[pad[0] : -pad[1] - 1, pad[2] : -pad[3] - 1]

    np.save("visualizations/segmentations_merge/merge_all_{}.npy".format(id), merge_all)

    merge_all_raw = np.copy(merge_all)
    merge_all_raw = post_process(merge_all_raw)
    merge_all_raw = post_process(merge_all_raw)

    # merge_all_raw = remove_gap(merge_all_raw)
    # merge_all_raw = process(merge_all_raw)
    # merge_all_raw = post_process(merge_all_raw)

    merge_all_raw = post_process_boundary(merge_all_raw)
    # # merge_all_raw = remove_float_seg(merge_all_raw)
    merge_all_raw = post_process_boundary(merge_all_raw)

    merge_all_raw = seq_name_seg(merge_all_raw)

    num_seg = np.unique(merge_all_raw)
    num_seg = np.asarray(num_seg)

    np.save(
        "visualizations/segmentations_merge/merge_all_post_{}.npy".format(id),
        merge_all_raw,
    )

    return merge_all_raw, merge_all


def merge_neighbor_row(tile1, tile2):
    n = 64  # 128
    w = tile1.shape[0]
    h = tile1.shape[1]

    bar1 = tile1[:, -80:-48]
    bar2 = tile2[:, 48:80]
    bar1 = bar1.flatten()
    bar2 = bar2.flatten()

    bar_pair_idx = np.zeros((np.amax(bar1) + 1, np.amax(bar2) + 1))

    for i in range(bar1.shape[0]):
        bar_pair_idx[bar1[i], bar2[i]] += 1

    for i in range(bar_pair_idx.shape[1]):
        max_id = np.argmax(bar_pair_idx[:, i])
        max_val = bar_pair_idx[max_id, i]

        bar_pair_idx[:, i] = 0
        bar_pair_idx[max_id, i] = max_val

    non_zero = np.nonzero(bar_pair_idx)
    non_zero = np.asarray(non_zero)

    thresh = 0  # ?

    final_pair = []

    for i in range(non_zero.shape[1]):
        count = bar_pair_idx[non_zero[0, i], non_zero[1, i]]
        if count > thresh:
            final_pair.append([non_zero[0, i], non_zero[1, i] + np.amax(tile1)])

    tile2 = tile2 + np.amax(tile1)

    for i in range(len(final_pair)):
        tile2[tile2 == final_pair[i][1]] = final_pair[i][0]

    merge_tile = np.concatenate((tile1[:, :-64], tile2[:, 64:]), axis=1)

    return merge_tile


def merge_col(tile1, tile2):
    w = tile1.shape[0]
    h = tile1.shape[1]

    bar1 = tile1[-80:-48, :]
    bar2 = tile2[48:80, :]
    bar1 = bar1.flatten()
    bar2 = bar2.flatten()

    bar_pair_idx = np.zeros((np.amax(bar1) + 1, np.amax(bar2) + 1))

    for i in range(bar1.shape[0]):
        bar_pair_idx[bar1[i], bar2[i]] += 1

    for i in range(bar_pair_idx.shape[1]):
        max_id = np.argmax(bar_pair_idx[:, i])
        max_val = bar_pair_idx[max_id, i]

        bar_pair_idx[:, i] = 0
        bar_pair_idx[max_id, i] = max_val

    non_zero = np.nonzero(bar_pair_idx)
    non_zero = np.asarray(non_zero)

    thresh = 0  # ?

    final_pair = []
    for i in range(non_zero.shape[1]):
        count = bar_pair_idx[non_zero[0, i], non_zero[1, i]]
        if count > thresh:
            final_pair.append([non_zero[0, i], non_zero[1, i] + np.amax(tile1)])

    tile2 = tile2 + np.amax(tile1)

    for i in range(len(final_pair)):
        tile2[tile2 == final_pair[i][1]] = final_pair[i][0]

    merge_tile = np.concatenate((tile1[:-64, :], tile2[64:, :]), axis=0)

    return merge_tile


from datetime import datetime


def viz_image(image, i=0):
    image = image.astype(np.uint8)

    image = cv2.resize(
        np.stack(
            [colors_256[image, 0], colors_256[image, 1], colors_256[image, 2]], axis=2
        ),
        (image.shape[1], image.shape[0]),
    )

    cv2.imwrite(
        "visualizations/check_new/" + "%d_%s.png" % (i, str(datetime.now())), image
    )


def process(seg, i=0):
    """
    Assign different colors to different segments
    """
    # viz_image(seg)
    seg_new = np.zeros_like(seg)

    for i in range(0, np.amax(seg) + 1):
        mask_seg = np.ma.masked_where(seg == i, seg).mask

        if mask_seg.ndim < 2:
            continue

        sep = measure.label(mask_seg, connectivity=1)
        maxx = np.amax(seg_new) * np.ma.masked_where(sep != 0, mask_seg).mask
        sep = sep + maxx
        seg_new = seg_new + sep

    return seg_new


def visualize(input, label, features, features1, semantic, semantic_pred, name, id):
    """
    This function performs postprocessing (dimensionality reduction and clustering) for a given network
    output. it also visualizes the resulted segmentation along with the original image and the ground truth
    segmentation and saves all the images locally.
    :param input: (3, h, w) ndarray containing rgb data as outputted by the costume datasets
    :param label: (h, w) or (1, h, w) ndarray with the ground truth segmentation
    :param features: (c, h, w) ndarray with the embedded pixels outputted by the network
    :param name: str with the current experiment name
    :param id: an identifier for the current image (for file saving purposes)
    :return: None. all the visualizations are saved locally
    """

    os.makedirs("visualizations/segmentations_merge/", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # visualization and evaluation
    pred_seg = features
    pred_seg1 = features1

    image = tensor_to_image(input)
    label = np.int64(label)
    semantic = np.int64(semantic)
    semantic_pred = np.int64(semantic_pred)

    new_viz, new_viz1 = merge_semantic_instance(semantic, semantic_pred, pred_seg)
    # new_viz, new_viz1 = semantic, semantic

    # change color for better visualization for semantic
    semantic[semantic == 2] = 11
    semantic_pred[semantic_pred == 2] = 11
    new_viz[new_viz == 2] = 11
    new_viz1[new_viz1 == 2] = 11

    semantic[semantic == 4] = 18
    semantic_pred[semantic_pred == 4] = 18
    new_viz[new_viz == 4] = 18
    new_viz1[new_viz1 == 4] = 18

    pred_seg = color_image(pred_seg)
    pred_seg1 = color_image(pred_seg1)
    label = color_image(label)
    semantic_pred = color_image(semantic_pred)
    semantic = color_image(semantic)
    new_viz = color_image(new_viz)
    new_viz1 = color_image(new_viz1)

    # blend image
    blend_pred = (pred_seg * 0.6 + image * 0.4).astype(np.uint8)
    blend_label = (label * 0.6 + image * 0.4).astype(np.uint8)

    p = 40  # pad size for visualization
    image = np.pad(
        image, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    pred_seg = np.pad(
        pred_seg, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    pred_seg1 = np.pad(
        pred_seg1, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    label = np.pad(
        label, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    blend_label = np.pad(
        blend_label, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    blend_pred = np.pad(
        blend_pred, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    new_viz = np.pad(
        new_viz, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    new_viz1 = np.pad(
        new_viz1, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    semantic_pred = np.pad(
        semantic_pred, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )
    semantic = np.pad(
        semantic, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
    )

    image0 = pred_seg
    # image1 = np.concatenate((image, label, pred_seg, blend_label, blend_pred), axis=1)
    image1 = np.concatenate(
        (
            image,
            label,
            pred_seg1,
            pred_seg,
            semantic,
            semantic_pred,
            new_viz,
            new_viz1,
            blend_label,
            blend_pred,
        ),
        axis=1,
    )
    image2 = np.concatenate((image, label, pred_seg, pred_seg1), axis=1)

    image1 = add_text1(image1)
    image2 = add_text2(image2)

    cv2.imwrite("visualizations/segmentations_merge/" + "result_%01d.png" % id, image0)
    cv2.imwrite(
        "visualizations/segmentations_merge/" + "all_result_%01d.png" % id, image1
    )
    cv2.imwrite(
        "visualizations/segmentations_merge/" + "all_results2_%01d.png" % id, image2
    )

    return


def color_image(image):
    image = image.astype(np.uint8)

    image = cv2.resize(
        np.stack(
            [colors_256[image, 0], colors_256[image, 1], colors_256[image, 2]], axis=2
        ),
        (image.shape[1], image.shape[0]),
    )

    return image


def post_process(seg):
    # merge small segmens with the majority of surrounding segments

    # num_seg = np.unique(seg)
    # num_seg = np.asarray(num_seg)

    unique, counts = np.unique(seg, return_counts=True)
    unique_dic = dict(zip(unique, counts))

    # pad for solving boundary issue
    seg_pad = np.pad(seg, [(1, 1), (1, 1)], mode="constant", constant_values=-1)

    s = sum(counts)
    threshold = 100  # int(s/3000)
    # threshold = 384 # 1024
    for key, value in unique_dic.items():
        neighbors = []
        neighbors_unique = []
        value_boundary = 0

        if value < threshold:
            result = np.where(seg_pad == key)
            x = list(result[0])
            y = list(result[1])

            # points = []
            for i in range(len(x)):
                # points.append([x[i], y[i]])
                neighbor = seg_pad[x[i] - 1 : x[i] + 2, y[i] - 1 : y[i] + 2]
                neighbor = neighbor.flatten()
                neighbor2 = neighbor
                neighbor = list(neighbor)

                neighbor_unique = np.unique(neighbor2)
                neighbor_unique = list(neighbor_unique)

                if key in neighbor:
                    neighbor = list(filter(lambda a: a != key, neighbor))

                if -1 in neighbor:
                    neighbor = list(filter(lambda a: a != -1, neighbor))

                # check if that pixel is boundary
                if len(neighbor):
                    value_boundary += 1

                neighbors.extend(neighbor)
                neighbors_unique.extend(neighbor_unique)

        if len(neighbors):
            # get most frequent element
            freq = max(set(neighbors), key=neighbors.count)

            num = neighbors_unique.count(freq)
            ratio = num / value_boundary

            if value < 99 or ratio > 0.9 or len(list(set(neighbors))) == 2:  # 0.9
                # change segment "key" to "freq"
                pass
                seg[seg == key] = freq

    return seg


def remove_float_seg(seg):
    # remove float segments in background

    unique, counts = np.unique(seg, return_counts=True)
    unique_dic = dict(zip(unique, counts))

    # pad for solving boundary issue
    seg_pad = np.pad(seg, [(1, 1), (1, 1)], mode="constant", constant_values=-1)

    for key, value in unique_dic.items():
        neighbors = []
        neighbors_unique = []
        value_boundary = 0

        result = np.where(seg_pad == key)
        x = list(result[0])
        y = list(result[1])

        # points = []
        for i in range(len(x)):
            # points.append([x[i], y[i]])
            neighbor = seg_pad[x[i] - 1 : x[i] + 2, y[i] - 1 : y[i] + 2]
            neighbor = neighbor.flatten()
            neighbor2 = neighbor
            neighbor = list(neighbor)

            neighbor_unique = np.unique(neighbor2)
            neighbor_unique = list(neighbor_unique)

            if key in neighbor:
                neighbor = list(filter(lambda a: a != key, neighbor))

            if -1 in neighbor:
                neighbor = list(filter(lambda a: a != -1, neighbor))

            # check if that pixel is boundary
            if len(neighbor):
                value_boundary += 1

            neighbors.extend(neighbor)
            neighbors_unique.extend(neighbor_unique)

        if len(neighbors):
            if set(neighbors) == set([0]):
                # change segment "key" to background
                seg[seg == key] = 0

    return seg


def post_process_boundary(seg):
    # to reserve 0 for background
    seg = seg + 1

    boundary_pixels = []
    # change all segments neighbor to image boundary as a background
    boundary_pixels.extend(seg[0, :])
    boundary_pixels.extend(seg[-1, :])
    boundary_pixels.extend(seg[:, 0])
    boundary_pixels.extend(seg[:, -1])

    boundary_pixels_unique = list(set(boundary_pixels))

    unique, counts = np.unique(boundary_pixels, return_counts=True)
    unique_dic = dict(zip(unique, counts))

    # pad for solving boundary issue
    seg_pad = np.pad(seg, [(1, 1), (1, 1)], mode="constant", constant_values=-1)
    counter = 0
    # not consider as a background pixel if boubdary of that segment is much larger than image boundary
    for key, value in unique_dic.items():
        # value = number of pixels for that segment in boundary of image
        # count number of surrounding pixels = value_boundary
        value_boundary = 0
        result = np.where(seg_pad == key)
        x = list(result[0])
        y = list(result[1])

        # points = []
        for i in range(len(x)):
            # points.append([x[i], y[i]])
            neighbor = seg_pad[x[i] - 1 : x[i] + 2, y[i] - 1 : y[i] + 2]
            neighbor = neighbor.flatten()
            neighbor = list(neighbor)

            if key in neighbor:
                neighbor = list(filter(lambda a: a != key, neighbor))

            if -1 in neighbor:
                neighbor = list(filter(lambda a: a != -1, neighbor))

            # check if that pixel is boundary
            if len(neighbor):
                value_boundary += 1

        # not consider as a background pixel if boubdary of that segment is much larger than image boundary
        if value_boundary != 0:
            ratio = value / value_boundary

            area_all = seg.shape[0] * seg.shape[1]
            seg_list = seg.flatten()
            seg_list = seg_list.tolist()

            area = seg_list.count(key)
            area_ratio = area_all / 1000
            # if area < area_ratio or ratio > 0.3: # 0.4:
            seg[seg == key] = 0

    return seg


def remove_gap(seg):
    # remove 1 pixel gap

    for i in range(1, seg.shape[0] - 1):
        for j in range(1, seg.shape[1] - 1):
            # 1 pixel gap
            if len(list(set([seg[i - 1, j], seg[i, j], seg[i + 1, j]]))) == 3:
                if seg[i, j - 1] == seg[i, j] and seg[i, j + 1] == seg[i, j]:
                    seg[i, j] = seg[i - 1, j]

            if len(list(set([seg[i, j - 1], seg[i, j], seg[i, j + 1]]))) == 3:
                if seg[i - 1, j] == seg[i, j] and seg[i + 1, j] == seg[i, j]:
                    seg[i, j] = seg[i, j - 1]

    return seg


def seq_name_seg(seg):
    # change seg names to sequential numbers starting from 0

    num_seg = np.unique(seg)
    num_seg = np.asarray(num_seg)

    for i in range(num_seg.shape[0]):
        seg[seg == num_seg[i]] = i

    return seg


def check_tiles(images, labels, segs, id):
    raise Exception
    # visualiz all cropped in a nice way!

    segs = segs.astype("uint8")

    os.makedirs("", exist_ok=True)

    h_num = segs.shape[0]
    w_num = segs.shape[1]
    p = 14

    tile_all = np.empty((0, 0))
    image_all = np.empty((0, 0))
    label_all = np.empty((0, 0))

    for h_i in range(h_num):
        # to assign first elements
        w_i = 0
        tile = segs[h_i, w_i, :, :]
        image = tensor_to_image(images[h_i, w_i, :, :])
        label = np.int64(labels[h_i, w_i, :, :])

        tile = color_image(tile)
        label = color_image(label)

        tile = np.pad(
            tile, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
        )
        image = np.pad(
            image, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
        )
        label = np.pad(
            label, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
        )

        tile_row = tile
        image_row = image
        label_row = label

        for w_i in range(1, w_num):
            tile = segs[h_i, w_i, :, :]
            image = tensor_to_image(images[h_i, w_i, :, :])
            label = np.int64(labels[h_i, w_i, :, :])

            tile = color_image(tile)
            label = color_image(label)

            tile = np.pad(
                tile, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
            )
            image = np.pad(
                image, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
            )
            label = np.pad(
                label, [(p, p), (p, p), (0, 0)], mode="constant", constant_values=1.0
            )

            tile_row = np.concatenate((tile_row, tile), axis=1)
            image_row = np.concatenate((image_row, image), axis=1)
            label_row = np.concatenate((label_row, label), axis=1)

        if h_i == 0:
            tile_all = tile_row
            image_all = image_row
            label_all = label_row
        else:
            tile_all = np.concatenate((tile_all, tile_row), axis=0)
            image_all = np.concatenate((image_all, image_row), axis=0)
            label_all = np.concatenate((label_all, label_row), axis=0)

    cv2.imwrite("" + "check_pred_{}.png".format(id), tile_all)
    cv2.imwrite("" + "check_image_{}.png".format(id), image_all)
    cv2.imwrite("" + "check_label_{}.png".format(id), label_all)

    return


def add_text1(image):
    # add text
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 2

    text = [
        "image",
        "instance_gt",
        "instance_pred_preprocessed",
        "instance_pred",
        "semantic_gt",
        "semantic_pred",
        "instance_pred + semantic_pred",
        "instance pred + semantic_gt",
        "image + instance_gt",
        "image + instance_pred",
    ]
    r = len(text)

    for i in range(r):
        x = int(image.shape[1] / (r * 2) + i * image.shape[1] / r) - 140
        # x = int(25 + i*image.shape[1]/r)

        org = (x, 30)
        text1 = text[i]
        image = cv2.putText(
            image, text1, org, font, fontScale, color, thickness, cv2.LINE_AA
        )

    return image


def add_text2(image2):
    # add text
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    return image2


def merge_semantic_instance(semantic, semantic_pred, instance_pred):
    r = 3
    weights = {0: 0, 1: 1, 2: r, 3: r, 4: r, 5: r, 6: r, 7: 1}

    # to don't have 0
    semantic_pred = semantic_pred + 1
    semantic = semantic + 1

    num_seg = np.unique(instance_pred)
    num_seg = np.asarray(num_seg)

    new_viz = np.zeros_like(instance_pred)
    new_viz1 = np.zeros_like(instance_pred)

    for i in range(num_seg.shape[0]):
        mask_seg = np.ma.masked_where(instance_pred == num_seg[i], instance_pred).mask

        label = mask_seg * semantic_pred
        label1 = mask_seg * semantic

        unique, counts = np.unique(label, return_counts=True)

        unique = np.asarray(unique)
        counts = np.asarray(counts)

        # add weighted
        for j in range(counts.shape[0]):
            counts[j] = counts[j] * weights[unique[j]]

        # sort_counts_index = np.argsort(counts)
        # index_max = sort_counts_index[-2]
        # maxx = unique[index_max]
        index_max = np.argmax(counts)
        maxx = unique[index_max]

        new_viz[instance_pred == num_seg[i]] = maxx

        # -----------gt

        unique, counts = np.unique(label1, return_counts=True)
        unique = np.asarray(unique)
        counts = np.asarray(counts)

        for j in range(counts.shape[0]):
            counts[j] = counts[j] * weights[unique[j]]

        index_max = np.argmax(counts)
        maxx = unique[index_max]

        new_viz1[instance_pred == num_seg[i]] = maxx

    new_viz = new_viz - 1
    new_viz1 = new_viz1 - 1

    return new_viz, new_viz1


# --------------method2: voting--------------------------


def save_data(
    pred_segs,
    pred_segs1,
    images,
    labels,
    image_full,
    label_full,
    semantic_full,
    semantic_pred_full,
    pad,
    idx,
):
    raise Exception

    path = ""
    path = path + "{}".format(idx)
    os.makedirs(path, exist_ok=True)

    np.save(path + "/pred_segs.npy", pred_segs)
    np.save(path + "/pred_segs1.npy", pred_segs1)
    np.save(path + "/images.npy", images)
    np.save(path + "/labels.npy", labels)
    np.save(path + "/image_full.npy", image_full)
    np.save(path + "/label_full.npy", label_full)
    np.save(path + "/semantic_full.npy", semantic_full)
    np.save(path + "/semantic_pred_full.npy", semantic_pred_full)
    np.save(path + "/pad.npy", pad)

    return


def find_xy_full(i, j, h_i, w_i, n):
    y = h_i * n + i
    x = w_i * n + j

    # print("check", y, x)

    return y, x


def merge_tiles_2(segs, id, pad, h, w):
    print("HEY Baby!")

    segs = segs.astype("uint32")

    os.makedirs("visualizations/segmentations_merge/", exist_ok=True)

    h_num = segs.shape[0]
    w_num = segs.shape[1]
    size_t = segs.shape[2]
    n = 16  # non_overlap_size
    h = h + pad[0] + pad[1]
    w = w + pad[2] + pad[3]

    print("h, w: ", h, w, size_t, pad, h_num, w_num)
    # full_index = np.zeros((h, w))
    full_index = np.zeros((h, w, 3, 3))

    seg_ones = np.zeros((h, w, 3, 3))
    seg_one = np.ones((size_t, size_t))
    seg_ones = seg_ones.astype("uint32")

    # h_num = 2
    # w_num = 2

    for h_i in range(h_num):
        print("*", h_i)
        for w_i in range(w_num):
            seg = segs[h_i, w_i, :, :]
            seg = seq_name_seg(seg)
            seg_pad = np.pad(seg, [(1, 1), (1, 1)], mode="constant", constant_values=-1)

            # h_i*n:h_i*n+256, w_i*n:w_i*n+256

            # for every pixel in segment
            for i in range(1, seg.shape[0] + 1):
                for j in range(1, seg.shape[1] + 1):
                    y, x = find_xy_full(i - 1, j - 1, h_i, w_i, n)
                    # print(x, y)
                    # seg_ones[y, x] += 1
                    neighbor_ones = np.ones((3, 3))
                    seg_ones[y, x, :, :] = seg_ones[y, x, :, :] + neighbor_ones

                    neighbor = seg_pad[i - 1 : i + 2, j - 1 : j + 2]
                    neighbor = neighbor.flatten()
                    # neighbor = np.delete(neighbor, 4)
                    main_pixel = seg_pad[i, j]
                    equal_neighbor_index = np.where(neighbor == main_pixel)
                    equal_neighbor_index = np.asarray(equal_neighbor_index)[0]
                    neighbor_final = np.zeros(9)
                    neighbor_final[equal_neighbor_index] = 1
                    neighbor_final = neighbor_final.reshape(3, 3)
                    neighbor_final = neighbor_final.astype("uint32")

                    full_index[y, x, :, :] = full_index[y, x, :, :] + neighbor_final
    # h = 150
    # w = 150
    sh = np.zeros((h, w))
    oo = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            sh[i, j] = np.mean(full_index[i, j, :, :])
            oo[i, j] = np.mean(seg_ones[i, j, :, :])

    plt.imshow(sh)
    plt.savefig("sh{}.png".format(id))
    # print(debug)
    plt.imshow(oo)
    plt.savefig("oo{}.png".format(id))

    np.save("full_index{}.npy".format(id), full_index)
    np.save("seg_ones{}.npy".format(id), seg_ones)
    np.save("sh{}.npy".format(id), sh)
    np.save("oo{}.npy".format(id), oo)

    print(full_index.shape)
    # seg_ones_major = (seg_ones+1)/2 # remove +1
    print(seg_ones.shape)

    segmented_index = np.ones((h, w)) * 2
    segmented_index = np.pad(
        segmented_index, [(1, 1), (1, 1)], mode="constant", constant_values=1
    )
    segmented_index = segmented_index.astype("uint32")
    full_index = full_index.astype("uint32")
    seg_ones = seg_ones.astype("uint32")

    return


def find_id(x, y, h, w, n, size_t, h_total, w_total):
    yy = h * n + y
    xx = w * n + x

    final_id = xx * w_total + yy

    return final_id


# -------------------------------------------------
# from scipy import ndimage
from scipy import ndimage as ndi
from skimage import io, img_as_bool, measure, morphology, segmentation


from scipy import ndimage


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


import sys

sys.setrecursionlimit(2000000000)


def floodFill(x, y, full_index, seg_ones, color_new):  # , segmented_index):
    # segmented_index2 = segmented_index[:]
    global segmented_index
    threshold = 15 / 16  # 0.99 #15/16 # 0.95
    h, w = segmented_index.shape

    # Base cases
    if (
        x < 0 or x >= h or y < 0 or y >= w
    ):  # or segmented_index[x, y] == color_new): # or screen[x][y] == newC):
        # print("Inside")
        return  # segmented_index

    # Replace the color at (x, y)
    # segmented_index[x, y] = color_new

    neighbors = full_index[x, y, :, :]
    ones = seg_ones[x, y, :, :]

    neighbor_ratio = neighbors / ones

    # threshold =  1 - 1/(np.sum(neighbors)-1)
    # threshold = ones - 2

    # Recur for neighbors based on major voting

    if neighbor_ratio[0, 1] >= threshold and segmented_index[x - 1, y] != color_new:
        # print("floodfill -1 0")

        if segmented_index[x - 1, y] != 0:
            segmented_index[segmented_index == segmented_index[x - 1, y]] = color_new
        else:
            segmented_index[x - 1, y] = color_new

            floodFill(x - 1, y, full_index, seg_ones, color_new)  # , segmented_index)

    if neighbor_ratio[1, 0] >= threshold and segmented_index[x, y - 1] != color_new:
        # if neighbors[1, 0] == np.amax(neighbors) and segmented_index[x, y-1] != color_new:
        if segmented_index[x, y - 1] != 0:
            segmented_index[segmented_index == segmented_index[x, y - 1]] = color_new
        else:
            segmented_index[x, y - 1] = color_new
            floodFill(x, y - 1, full_index, seg_ones, color_new)  # , segmented_index)

    # print("END")
    return


def auto_canny(image, sigma=1):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def final_seg_merge(segs, idx, pad, h, w, n, configs):
    segs = segs.astype("uint32")

    h_num = segs.shape[0]
    w_num = segs.shape[1]
    size_t = segs.shape[2]
    # n = 16 # non_overlap_size
    h = h + pad[0] + pad[1]
    w = w + pad[2] + pad[3]

    print("h, w: ", h, w, size_t, pad, h_num, w_num)
    full_index = np.zeros((h, w, 3, 3))
    full_index = full_index.astype("uint32")

    seg_ones = np.zeros((h, w, 3, 3))
    seg_one = np.ones((size_t, size_t))
    seg_ones = seg_ones.astype("uint32")

    # h_num = 5
    # w_num = 5

    for h_i in range(h_num):
        print("*", h_i)
        for w_i in range(w_num):
            seg = segs[h_i, w_i, :, :]
            seg = seq_name_seg(seg)
            seg_pad = np.pad(seg, [(1, 1), (1, 1)], mode="constant", constant_values=-1)
            seg_one_pad = np.pad(
                seg_one, [(1, 1), (1, 1)], mode="constant", constant_values=0
            )

            # h_i*n:h_i*n+256, w_i*n:w_i*n+256

            # for every pixel in segment
            for i in range(1, seg.shape[0] + 1):
                for j in range(1, seg.shape[1] + 1):
                    y, x = find_xy_full(i - 1, j - 1, h_i, w_i, n)
                    # print(x, y)
                    # seg_ones[y, x] += 1

                    neighbor = seg_pad[i - 1 : i + 2, j - 1 : j + 2]
                    neighbor = neighbor.flatten()
                    # neighbor = np.delete(neighbor, 4)
                    main_pixel = seg_pad[i, j]
                    equal_neighbor_index = np.where(neighbor == main_pixel)
                    equal_neighbor_index = np.asarray(equal_neighbor_index)[0]
                    neighbor_final = np.zeros(9)
                    neighbor_final[equal_neighbor_index] = 1
                    neighbor_final = neighbor_final.reshape(3, 3)
                    neighbor_final = neighbor_final.astype("uint32")
                    full_index[y, x, :, :] = full_index[y, x, :, :] + neighbor_final

                    neighbor = seg_one_pad[i - 1 : i + 2, j - 1 : j + 2]
                    neighbor_final = neighbor
                    neighbor_final = neighbor_final.astype("uint32")
                    seg_ones[y, x, :, :] = seg_ones[y, x, :, :] + neighbor_final

    # remove padding
    # final_index = final_index[pad[0]:-pad[1], pad[2]:-pad[3]]
    full_index1 = full_index[pad[0] + 1 : -pad[1] + 1, pad[2] + 1 : -pad[3] + 1, :, :]
    seg_ones1 = seg_ones[pad[0] + 1 : -pad[1] + 1, pad[2] + 1 : -pad[3] + 1, :, :]

    # NOTE optional save files
    # np.save(configs.base_dir + '/Data/full_index_{}.npy'.format(idx), full_index1)
    # np.save(configs.base_dir + '/Data/seg_ones_{}.npy'.format(idx), seg_ones1)

    # full_index = np.load("Data/full_index_{}.npy".format(idx))
    # seg_ones = np.load("Data/seg_ones_{}.npy".format(idx))

    # h = (h_num-1)*n+256
    # w = (w_num-1)*n+256
    full_index_mean = np.zeros((h, w))
    seg_ones_mean = np.zeros((h, w))
    final_index = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            full_index_mean[i, j] = np.mean(full_index[i, j, :, :])
            seg_ones_mean[i, j] = np.mean(seg_ones[i, j, :, :])

            if seg_ones_mean[i, j] != 0:
                final_index[i, j] = full_index_mean[i, j] / seg_ones_mean[i, j]
            else:
                final_index[i, j] = 0

    # np.save('Data/full_index_mean{}.npy'.format(idx), full_index_mean)
    # np.save('Data/seg_ones_mean{}.npy'.format(idx), seg_ones_mean)
    # remove padding
    final_index = final_index[pad[0] + 1 : -pad[1] + 1, pad[2] + 1 : -pad[3] + 1]
    np.save(
        configs.base_dir + "/preprocess/boundary_pred/{}.npy".format(idx), final_index
    )

    # plt.imshow(final_index)
    # plt.savefig("Data/final_index_{}.svg".format(idx), format='svg', dpi=1200)
    final_index_cv = np.stack((final_index,) * 3, axis=-1)
    final_index_cv = final_index_cv * 255
    cv2.imwrite(
        configs.base_dir + "/preprocess/boundary_pred/{}.png".format(idx),
        final_index_cv,
    )

    return final_index


def add_text10(image):
    # add text
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 2

    text = [
        "image",
        "instance_gt",
        "boundary",
        "instance_pred_boundary",
        "semantic_gt",
        "semantic_pred",
        "instance_pred + semantic_pred",
        "instance pred + semantic_gt",
        "image + instance_gt",
        "image + instance_pred",
    ]
    r = len(text)

    for i in range(r):
        x = int(image.shape[1] / (r * 2) + i * image.shape[1] / r) - 140
        # x = int(25 + i*image.shape[1]/r)

        org = (x, 30)
        text1 = text[i]
        image = cv2.putText(
            image, text1, org, font, fontScale, color, thickness, cv2.LINE_AA
        )

    return image


def post_process_boundary_black(seg):
    # to reserve 0 for background
    seg = seg + 1

    unique = np.unique(seg[0:4, :])
    unique = list(np.asarray(unique))
    unique.extend(list(np.asarray(np.unique(seg[-5:-1, :]))))
    unique = list(np.asarray(unique))
    unique.extend(list(np.asarray(np.unique(seg[:, -5:-1]))))
    unique = list(np.asarray(unique))
    unique.extend(list(np.asarray(np.unique(seg[:, 0:4]))))
    unique = list(np.asarray(unique))

    for i in unique:
        seg[seg == i] = 0

    return seg
