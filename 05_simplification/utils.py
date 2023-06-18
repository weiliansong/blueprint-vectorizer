import os
import json
import glob
import shutil
import argparse
import numpy as np

# import cv2
# import torch
import math
import matplotlib.pyplot as plt
from itertools import groupby
from tqdm import tqdm
from ruamel.yaml import YAML
from PIL import Image
from skimage import measure
from skimage.morphology import erosion, dilation, square
from scipy.ndimage import gaussian_filter

# import torchvision.transforms as transforms


GA_ROOT = "../data/"
PREPROCESS_ROOT = GA_ROOT + "preprocess/"
SPLITS_ROOT = GA_ROOT + "splits/"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", dest="method", action="store", type=str)

    parser.add_argument(
        "--num_threads", dest="num_threads", action="store", type=int, default=1
    )

    parser.add_argument("--floorplan_id", dest="floorplan_id", action="store", type=str)

    parser.add_argument(
        "--vertex_weight", dest="vertex_weight", action="store", type=float, default=1
    )

    parser.add_argument(
        "--hparam_f", dest="hparam_f", action="store", type=str, default=""
    )

    parser.add_argument("--restart", dest="restart", action="store_true", default=False)

    parser.add_argument(
        "--experiment_root", dest="experiment_root", action="store", type=str
    )

    args = parser.parse_args()

    return args


def parse_config(config_path):
    yaml = YAML(typ="safe")
    config = yaml.load(open(config_path, "r"))

    return config


def ensure_dir(dir_path, remove=False):
    if remove and os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def remove_dir(path_name):
    if os.path.exists(os.path.dirname(path_name)):
        shutil.rmtree(os.path.dirname(path_name))


def get_fp_bbox(segmentation, margin_size=32):
    img_h, img_w = segmentation.shape
    yy, xx = np.nonzero(segmentation > 0)

    minx = int(np.min(xx))
    miny = int(np.min(yy))
    maxx = int(np.max(xx))
    maxy = int(np.max(yy))

    # add small padding
    miny = max(miny - margin_size, 0)
    minx = max(minx - margin_size, 0)
    maxy = min(maxy + margin_size, img_h)
    maxx = min(maxx + margin_size, img_w)

    return miny, minx, maxy, maxx


def circularly_identical(list1, list2):
    _list1 = ["|%d|" % x for x in list1]
    _list2 = ["|%d|" % x for x in list2]
    return (" ".join(_list1) in " ".join(_list2 * 2)) or (
        " ".join(_list1[::-1]) in " ".join(_list2 * 2)
    )


def find_all_identical_keys(query, keys):
    identical_keys = []

    if keys:
        for key in keys:
            if circularly_identical(query, key) and (len(query) == len(key)):
                identical_keys.append(key)

    return identical_keys


# since input is a circular list, we also check the two ends as well
def remove_consecutive_duplicates(ring):
    if not len(ring):
        return ring

    if len(np.unique(ring)) == 1:
        return [
            ring[0],
        ]

    _ring = [x[0] for x in groupby(ring)]

    if _ring[0] == _ring[-1]:
        return _ring[:-1]
    else:
        return _ring


# circularly shift the list of semantic mappings so that the smallest is at
# the beginning
def circular_shift(sem_ring):
    min_idx = np.argmin(sem_ring)
    shifted = sem_ring[min_idx:] + sem_ring[:min_idx]
    return shifted


def get_sem_ring_mapping():
    json_files = glob.glob("./input/sem_ring/*.json")

    all_sem_rings = []
    for json_f in json_files:
        fp_id = json_f.split("/")[-1].split(".")[0]

        with open(json_f, "r") as f:
            for (instance_id, instance_sem), instance_ring in json.load(f):
                all_sem_rings.append(
                    ((fp_id, instance_id, instance_sem), instance_ring)
                )

    print("Computing necessary mappings")
    count_mapping = {3: {}, 4: {}, 5: {}, 7: {}}
    example_mapping = {3: {}, 4: {}, 5: {}, 7: {}}

    for (fp_id, instance_id, instance_sem), instance_ring in tqdm(all_sem_rings):
        # skip bg and walls and rooms
        if instance_sem in [0, 1, 2, 6]:
            continue

        sem_ring = [sem for (edge, sem, instance_id) in instance_ring]
        sem_ring = remove_consecutive_duplicates(sem_ring)
        sem_ring = tuple(circular_shift(sem_ring))

        keys = find_all_identical_keys(sem_ring, count_mapping[instance_sem].keys())

        if not len(keys):
            keys = [
                sem_ring,
            ]
            count_mapping[instance_sem][sem_ring] = 0
            example_mapping[instance_sem][sem_ring] = []

        for key in keys:
            count_mapping[instance_sem][key] += 1
            example_mapping[instance_sem][key].append((fp_id, instance_id))

    return count_mapping, example_mapping


# get all the predicted masks, including boundary, instance, and semantic
def get_pred_data(fp_id, sigma=1):
    boundary = np.load("./input/boundary/final_index{}.npy".format(fp_id))
    semantic = np.load("./input/pred_semantic/{}.npy".format(fp_id))
    # instance = np.load('./input/pred_instance/{}.npy'.format(fp_id))

    assert boundary.shape == semantic.shape

    # NOTE for now, get instance from semantic mask
    instance = measure.label(semantic, background=0)
    # semantic = remove_holes(instance, semantic)
    # instance = measure.label(semantic, background=0)

    # invert boundary image, where 1 is high edge probability
    boundary = 1 - boundary

    # also blur the boundary a bit
    boundary = gaussian_filter(boundary, sigma=sigma)

    # stack instance and semantic masks so it's easier to pass around
    segmentation = np.stack([instance, semantic], axis=0)

    return segmentation, boundary


# remove small bits in semantic floorplan
def remove_holes(all_instance_mask, all_semantic_mask):
    cleaned_semantic_mask = all_semantic_mask.copy()

    for instance_id in np.unique(all_instance_mask):
        all_neighbor_sems = semantic_adj_list.get_sem_adj_list(
            instance_id, all_instance_mask, all_semantic_mask
        )

        if (len(all_neighbor_sems) == 1) and (all_neighbor_sems[0][1] == 6):
            instance_mask = all_instance_mask == instance_id
            cleaned_semantic_mask[instance_mask] = all_neighbor_sems[0][1]

            if False:
                fig, [ax1, ax2] = plt.subplots(ncols=2)

                ax1.imshow(all_semantic_mask, cmap="nipy_spectral")
                ax2.imshow(cleaned_semantic_mask, cmap="nipy_spectral")
                ax2.imshow(instance_mask, cmap="hot", alpha=0.8)

                ax1.set_axis_off()
                ax2.set_axis_off()

                plt.tight_layout()
                plt.show()
                plt.close()

    return cleaned_semantic_mask


# get GT image, instance, and semantic masks
def get_gt_data(fp_id):
    image = np.array(Image.open("./input/fp_img/%s.jpg" % fp_id))
    semantic = np.load("./input/gt_semantic/{}.npy".format(fp_id))
    instance = np.load("./input/gt_instance/{}.npy".format(fp_id))

    assert image.shape == semantic.shape == instance.shape

    # map symbol class to 7
    semantic[semantic == 24] = 7

    # stack instance and semantic masks so it's easier to pass around
    segmentation = np.stack([instance, semantic], axis=0)

    return image, segmentation


# alpha: threshold for finding the boundary
# beta: maximum area for a segment to merge with neighbors
def get_segmentation(fp_id, alpha=0.9, beta=10, k=3, sigma=1):
    # load boundary map
    boundary = np.load("./input/boundary/final_index{}.npy".format(fp_id))
    h, w = boundary.shape

    # apply treshold
    thresholded = np.copy(boundary)
    thresholded[boundary > alpha] = 1
    thresholded[boundary <= alpha] = 0

    # find connected components and assign different color to each segment
    segmentation = measure.label(thresholded, connectivity=2)

    # remove 1 pixel boundaries
    segmentation = dilation((segmentation), square(k))
    segmentation = remove_boundaries(segmentation)

    # apply post processing to handle small regions
    segmentation = remove_small_seg(segmentation, beta)
    segmentation = seq_name_seg(segmentation)
    segmentation = post_process_bg_black(segmentation)

    # invert boundary image, where 1 is high edge probability
    boundary = 1 - boundary

    # also blur the boundary a bit
    boundary = gaussian_filter(boundary, sigma=sigma)

    return segmentation, boundary


def load_and_segment(idx, threshold1, threshold2, id_path, image_path, boundary_path):

    img_path = image_path + idx + ".jpg"
    image_full = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    image_full = tensor_to_image(image_full)
    cv2.imwrite("result/image_{}.png".format(idx), image_full)

    # probabilities boundaries
    final_index = np.load(boundary_path + "final_index{}.npy".format(idx))

    final_index_image = np.stack((final_index,) * 3, axis=-1) * 255
    cv2.imwrite("result/final_index_{}.png".format(idx), final_index_image)

    image = np.copy(final_index)
    h, w = image.shape

    # find edges
    # apply treshold
    image[image > threshold1] = 1
    image[image <= threshold1] = 0

    plt.imshow(image)
    plt.savefig("result/boundaries_{}.png".format(idx), dpi=1200)

    # find connected components and assign different color to each segment
    predict_segmentation = measure.label(image, connectivity=2)
    # remove 1 pixel boundaries
    k = 3
    predict_segmentation = dilation((predict_segmentation), square(k))

    predict_segmentation = remove_boundaries(predict_segmentation)

    # apply post processing to handle small regions
    predict_segmentation = remove_small_seg(predict_segmentation, threshold2)

    predict_segmentation = seq_name_seg(predict_segmentation)
    predict_segmentation = post_process_bg_black(predict_segmentation)

    np.save("result/predict_segmentation_{}.npy".format(idx), predict_segmentation)

    predict_segmentation_color = color_image(predict_segmentation)
    cv2.imwrite(
        "result/segments_boundary_{}.png".format(idx), predict_segmentation_color
    )

    return predict_segmentation, final_index


def dot(vA, vB):
    return vA[0] * vB[0] + vA[1] * vB[1]


def ang(lineA, lineB):
    # Get nicer vector form
    vA = lineA.tolist()
    vB = lineB.tolist()
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA) ** 0.5
    magB = dot(vB, vB) ** 0.5
    # Get cosine value
    cos_ = dot_prod / magA / magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod / magB / magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle) % 360

    if ang_deg - 180 >= 0:
        # As in if statement
        return 360 - ang_deg
    else:

        return ang_deg


def color_image(image):
    image = image.astype(np.uint8)

    image = cv2.resize(
        np.stack(
            [colors_256[image, 0], colors_256[image, 1], colors_256[image, 2]], axis=2
        ),
        (image.shape[1], image.shape[0]),
    )

    return image


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


def tensor_to_image(image):
    image = torch.from_numpy(image)  # change
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


# Tensor_to_Image = transforms.Compose([
#     transforms.ToPILImage()
# ])

# functions to process boundary image to segments:


def post_process_bg_black(seg):
    # return 0 in background and 1 in boundaries
    # to be robust use 4 pixel boub=

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


def process(seg, i=0):
    """
    Assign different colors to different segments
    """
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


def seq_name_seg(seg):
    # change seg names to sequential numbers starting from 0

    num_seg = np.unique(seg)
    num_seg = np.asarray(num_seg)

    for i in range(num_seg.shape[0]):
        seg[seg == num_seg[i]] = i

    return seg


def remove_small_seg(seg, threshold=10):

    seg = process(seg)
    # merge small segmens with the majority of surrounding segments
    h, w = seg.shape

    unique, counts = np.unique(seg, return_counts=True)

    unique_dic = dict(zip(unique, counts))

    # pad for solving boundary issue
    seg_pad = np.pad(seg, [(1, 1), (1, 1)], mode="constant", constant_values=-1)

    # s = sum(counts)
    # threshold  = ((h*w)/20000) # 100
    # threshold = 10
    # print("threshold", threshold)

    for key, value in unique_dic.items():
        neighbors = []
        neighbors_unique = []
        value_boundary = 0

        if value < threshold and key != -1:
            result = np.where(seg_pad == key)
            x = list(result[0])
            y = list(result[1])

            for i in range(len(x)):
                neighbor = seg_pad[x[i] - 1 : x[i] + 2, y[i] - 1 : y[i] + 2]
                neighbor = neighbor.flatten()
                neighbor2 = neighbor
                neighbor = list(neighbor)

                neighbor_unique = np.unique(neighbor2)
                neighbor_unique = list(neighbor_unique)

                if key in neighbor:
                    neighbor = list(filter(lambda a: a != key, neighbor))

                # if -1 in neighbor:
                #     neighbor = list(filter(lambda a: a != -1, neighbor))

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

                # if value < 99 or ratio > 0.9 or len(list(set(neighbors))) == 2: # 0.9
                #     # change segment "key" to "freq"
                #     # pass
                seg[seg == key] = freq

    seg = process(seg)

    return seg


def remove_boundaries(seg):

    # change all 0 pixels (boundaries) to the most repetetive neighbor color
    seg_pad = np.pad(seg, [(1, 1), (1, 1)], mode="constant", constant_values=1)

    boundaries = np.where(seg_pad == 0)
    x = list(boundaries[0])
    y = list(boundaries[1])

    patience = 0

    while len(x) > 0:
        for i in range(len(x)):

            neighbor = seg_pad[x[i] - 1 : x[i] + 2, y[i] - 1 : y[i] + 2]
            neighbor = neighbor.flatten()
            neighbor = np.delete(neighbor, 4)

            if patience == 1:
                neighbor = np.delete(neighbor, np.argwhere(neighbor == 0))
                patience = 0

            # get most frequent element
            unique, counts = np.unique(neighbor, return_counts=True)

            unique = np.asarray(unique)
            counts = np.asarray(counts)

            id_max_all = np.argwhere(counts == np.amax(counts))
            id_max_all = id_max_all.flatten().tolist()

            # choose color different from boundary color
            if len(id_max_all) > 1:
                if unique[id_max_all[0]] == 0:
                    id_max = id_max_all[1]
                else:
                    id_max = id_max_all[0]

            else:
                id_max = id_max_all[0]

            freq = unique[id_max]
            seg_pad[x[i], y[i]] = freq

        len_x_past = len(x)

        boundaries = np.where(seg_pad == 0)
        x = list(boundaries[0])
        y = list(boundaries[1])

        if len(x) == len_x_past:
            patience = 1

    seg = seg_pad[1:-1, 1:-1]

    return seg
