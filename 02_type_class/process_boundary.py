import numpy as np

from skimage import measure
from skimage.morphology import erosion, dilation, square

import paths


def load_and_segment(fp_id, threshold1=0.95, threshold2=10):
    # probabilities boundaries
    final_index = np.load(paths.PRED_BOUNDARY_ROOT + "%s.npy" % fp_id)

    image = np.copy(final_index)
    h, w = image.shape

    # apply treshold
    image[image > threshold1] = 1
    image[image <= threshold1] = 0

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

    return predict_segmentation


# functions to process boundary image to segments:
def post_process_bg_black(seg):
    # return 0 in background
    # to be robust use 4 pixel boundary

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
    Assign different colors to different segments (which is not connected)
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
