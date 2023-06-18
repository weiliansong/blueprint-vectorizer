import os
import glob
import copy
import json
import time
import pickle
import matplotlib
import multiprocessing as mp

from multiprocessing import Pool

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from skimage import measure
from contextlib import closing
from matplotlib.colors import ListedColormap
from skimage.morphology import square
from skimage.morphology import binary_dilation
from networkx.readwrite import json_graph

import utils
import parallel

# from timer import Timer
# from cost import cost_calculation


# random colormap for instance segmentation visualization
random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))

# qualitative visualization cmap
my_colors = {
    "bg": (255, 255, 255, 255),  # white
    "outer": (223, 132, 224, 255),  # purple
    "inner": (84, 135, 255, 255),  # blue
    "window": (255, 170, 84, 255),  # orange
    "door": (101, 255, 84, 255),  # green
    "frame": (255, 246, 84, 255),  # yellow
    "room": (230, 230, 230, 255),  # gray
    "symbol": (255, 87, 87, 255),  # red
}
colors = np.array(list(my_colors.values())) / 255.0
my_cmap = ListedColormap(colors)


def bbox_crop(arr, bbox, pad=10, return_bbox=False):
    mini, minj, maxi, maxj = bbox

    if len(arr.shape) == 2:
        h, w = arr.shape
    elif len(arr.shape) == 3:
        _, h, w = arr.shape
    else:
        raise Exception("Cannot handle cropping of this shape")

    if pad:
        shape = arr.shape
        mini = max(0, mini - pad)
        minj = max(0, minj - pad)
        maxi = min(h, maxi + pad)
        maxj = min(w, maxj + pad)

    if len(arr.shape) == 2:
        new_arr = arr[mini:maxi, minj:maxj]
    elif len(arr.shape) == 3:
        new_arr = arr[:, mini:maxi, minj:maxj]
    else:
        raise Exception("Cannot handle cropping of this shape")

    if return_bbox:
        return new_arr, (mini, minj, maxi, maxj)
    else:
        return new_arr


def bbox_overlap(l1, r1, l2, r2):
    # If one rectangle is on left side of other
    if l1[0] >= r2[0] or l2[0] >= r1[0]:
        return False

    # If one rectangle is above other
    if l1[1] <= r2[1] or l2[1] <= r1[1]:
        return False

    return True


def get_turn(graph, prev_node, curr_node, next_node):
    # turn coordinates into XY frame
    next_y, next_x = graph.nodes[next_node]["ij"]
    curr_y, curr_x = graph.nodes[curr_node]["ij"]
    prev_y, prev_x = graph.nodes[prev_node]["ij"]

    # make sure only one of the differences is 0
    diff_x = next_x - curr_x
    diff_y = next_y - curr_y
    assert (diff_x + diff_y != 0) and (diff_x * diff_y == 0)

    angle = np.arctan2(next_y - curr_y, next_x - curr_x) - np.arctan2(
        prev_y - curr_y, prev_x - curr_x
    )
    angle = np.degrees(angle)

    if angle in [-90, 270]:
        return "L"
    elif angle in [-270, 90]:
        return "R"
    elif abs(angle) == 180:
        return "S"  # "S" stands for straight
    else:
        raise Exception("Unknown angle %f" % angle)


def get_edges(graph):
    raw_edges = set(graph.edges)
    desired_edges = []

    while len(raw_edges):
        start, end = raw_edges.pop()
        edge = [start, end]
        prev_node = start
        curr_node = end

        while True:
            moved = False

            for next_node in graph[curr_node]:
                if next_node == prev_node:
                    continue

                elif get_turn(graph, prev_node, curr_node, next_node) == "S":
                    raw_edges.discard((curr_node, next_node))
                    raw_edges.discard((next_node, curr_node))
                    prev_node = curr_node
                    curr_node = next_node
                    edge.append(next_node)
                    moved = True
                    break

            if not moved:
                break

        prev_node = end
        curr_node = start

        while True:
            moved = False

            for next_node in graph[curr_node]:
                if next_node == prev_node:
                    continue

                elif get_turn(graph, prev_node, curr_node, next_node) == "S":
                    raw_edges.discard((curr_node, next_node))
                    raw_edges.discard((next_node, curr_node))
                    prev_node = curr_node
                    curr_node = next_node
                    edge.insert(0, next_node)
                    moved = True
                    break

            if not moved:
                break

        desired_edges.append(edge)

    return desired_edges


# helper function to split a chain of nodes into list of line segments
# in between nodes, "None" is used to separate the lines
def split_chain(chain):
    line = []

    for node in chain:
        if node != None:  # delimiter is None
            line.append(node)

        elif line:
            yield line
            line = [
                line[-1],
            ]


def vis_chain(chain, title):
    xx = []
    yy = []
    for node in chain:
        if node != None:
            y, x = graph.nodes[node]["ij"]
            xx.append(x)
            yy.append(y)

    plt.plot(xx, yy, "-")
    plt.plot(xx[1:], yy[1:], "o")
    plt.title(title)
    plt.show()
    plt.close()


def get_edge_length(v1, v2):
    i1, j1 = v1
    i2, j2 = v2

    assert (i1 == i2) or (j1 == j2)

    return max(abs(i2 - i1), abs(j2 - j1))


def get_all_edge_collapses(graph, segmentation, hparams):
    # all_edges = get_edges(graph)
    all_edges = list(set(graph.edges))

    # compute all the possible shifts we can do
    all_shift_params = []

    for edge in all_edges:
        start = edge[0]
        end = edge[-1]

        v1 = graph.nodes[start]["ij"]
        v2 = graph.nodes[end]["ij"]
        if get_edge_length(v1, v2) > hparams["short_edge_threshold"]:
            continue

        # no matter the orientation, the effects are limited to a bbox
        start_neighbors = np.array([graph.nodes[n]["ij"] for n in graph[start]])
        end_neighbors = np.array([graph.nodes[n]["ij"] for n in graph[end]])
        edge_coords = np.array([graph.nodes[n]["ij"] for n in edge])

        all_coords = np.concatenate([start_neighbors, end_neighbors, edge_coords])
        mini = int(all_coords[:, 0].min())
        minj = int(all_coords[:, 1].min())
        maxi = int(all_coords[:, 0].max())
        maxj = int(all_coords[:, 1].max())

        # check how the middle line is oriented, either horizontal or vertical
        edge_vector = edge_coords[-1] - edge_coords[0]

        if edge_vector[0] == 0 and edge_vector[1] != 0:
            orientation = "H"  # "H" stands for horizontal
        elif edge_vector[0] != 0 and edge_vector[1] == 0:
            orientation = "V"  # "V" stands for vertical
        else:
            raise Exception("Unknown middle line orientation")

        # we want to shift the minimum amount of distance
        all_shifts = []

        start_coord = np.array(graph.nodes[start]["ij"])
        for neighbor_node in graph[start]:
            if neighbor_node == end:
                continue
            if get_turn(graph, end, start, neighbor_node) == "S":
                continue

            neighbor_coord = np.array(graph.nodes[neighbor_node]["ij"])
            all_shifts.append(neighbor_coord - start_coord)

        end_coord = np.array(graph.nodes[end]["ij"])
        for neighbor_node in graph[end]:
            if neighbor_node == start:
                continue
            if get_turn(graph, start, end, neighbor_node) == "S":
                continue

            neighbor_coord = np.array(graph.nodes[neighbor_node]["ij"])
            all_shifts.append(neighbor_coord - end_coord)

        shift_distance = float("inf")

        if orientation == "H":
            for candidate_shift in all_shifts:
                assert (candidate_shift[1] == 0) and (candidate_shift[0] != 0)
                if abs(candidate_shift[0]) < abs(shift_distance):
                    shift_distance = candidate_shift[0]

        elif orientation == "V":
            for candidate_shift in all_shifts:
                assert (candidate_shift[0] == 0) and (candidate_shift[1] != 0)
                if abs(candidate_shift[1]) < abs(shift_distance):
                    shift_distance = candidate_shift[1]

        else:
            raise Exception("Unknown orientation")

        assert shift_distance < 1000

        if False:
            plt.imshow(segmentation[0], cmap=random_cmap)

            # plot boundary
            xx = [minj + 0.5, maxj + 0.5, maxj + 0.5, minj + 0.5, minj + 0.5]
            yy = [mini + 0.5, mini + 0.5, maxi + 0.5, maxi + 0.5, mini + 0.5]
            plt.plot(xx, yy, "-c")

            # plot this edge
            xx = [p[1] + 0.5 for p in edge_coords]
            yy = [p[0] + 0.5 for p in edge_coords]
            plt.plot(xx, yy, "-ok")

            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()

        shift_params = {
            "action": "edge_shift",
            "line": edge_coords.tolist(),
            "bbox": [mini, minj, maxi, maxj],
            "orientation": orientation,
            "distance": shift_distance,
        }

        all_shift_params.append(shift_params)

    return all_shift_params


# NOTE that segmentation is a stack of instance and semantic masks
# so you see segmentation[0], that's grabbing instance mask only
def shift_edge(segmentation, params, hparams, small_crop=False, vis=False):
    line_coords = np.array(params["line"])

    if params["orientation"] == "H":
        shift_by = np.array([params["distance"], 0])
    elif params["orientation"] == "V":
        shift_by = np.array([0, params["distance"]])
    else:
        raise Exception("Unknown orientation")

    # the +1 is kind of sketchy feeling, but it works
    replace_pts = np.concatenate([line_coords + shift_by, line_coords])
    replace_mini = replace_pts[:, 0].min() + 1
    replace_minj = replace_pts[:, 1].min() + 1
    replace_maxi = replace_pts[:, 0].max() + 1
    replace_maxj = replace_pts[:, 1].max() + 1

    copy_pts = np.concatenate([line_coords - np.sign(shift_by), line_coords])
    copy_mini = copy_pts[:, 0].min() + 1
    copy_minj = copy_pts[:, 1].min() + 1
    copy_maxi = copy_pts[:, 0].max() + 1
    copy_maxj = copy_pts[:, 1].max() + 1

    # find what we need to replace with
    copied_seg = segmentation[:, copy_mini:copy_maxi, copy_minj:copy_maxj]

    # repeat it to the correct size
    repeat_axis = 1 if (params["orientation"] == "H") else 2
    copied_seg = np.repeat(copied_seg, abs(params["distance"]), axis=repeat_axis)

    # shift the edge by changing colors
    if small_crop:
        crop_segmentation, crop_bbox = bbox_crop(
            segmentation, params["bbox"], return_bbox=True
        )
        crop_mini, crop_minj, crop_maxi, crop_maxj = crop_bbox

        replace_mini -= crop_mini
        replace_minj -= crop_minj
        replace_maxi -= crop_mini
        replace_maxj -= crop_minj

        new_segmentation = np.copy(crop_segmentation)
        new_segmentation[
            :, replace_mini:replace_maxi, replace_minj:replace_maxj
        ] = copied_seg

        # don't do the action if it ends up merging or splitting a region
        if not hparams["allow_merge_regions"]:
            tmp_segmentation = measure.label(new_segmentation[0], connectivity=2)
            old_segmentation = bbox_crop(segmentation, crop_bbox, pad=0)
            if len(np.unique(old_segmentation[0])) != len(
                np.unique(tmp_segmentation)
            ) or len(np.unique(old_segmentation[0])) != len(
                np.unique(new_segmentation[0])
            ):
                new_segmentation = old_segmentation

        # visualize the action
        if False:
            fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

            ax1.imshow(old_segmentation[0], cmap=random_cmap)

            replace_xx = (
                np.array(
                    [
                        replace_minj,
                        replace_maxj,
                        replace_maxj,
                        replace_minj,
                        replace_minj,
                    ]
                )
                - 0.5
            )
            replace_yy = (
                np.array(
                    [
                        replace_mini,
                        replace_mini,
                        replace_maxi,
                        replace_maxi,
                        replace_mini,
                    ]
                )
                - 0.5
            )
            ax1.plot(replace_xx, replace_yy, "-or")

            copy_xx = (
                np.array([copy_minj, copy_maxj, copy_maxj, copy_minj, copy_minj])
                - 0.5
                - crop_minj
            )
            copy_yy = (
                np.array([copy_mini, copy_mini, copy_maxi, copy_maxi, copy_mini])
                - 0.5
                - crop_mini
            )
            ax1.plot(copy_xx, copy_yy, "-ob")

            # visualize the segmentation after the shift
            ax2.imshow(new_segmentation[0], cmap=random_cmap)

            replace_xx = (
                np.array(
                    [
                        replace_minj,
                        replace_maxj,
                        replace_maxj,
                        replace_minj,
                        replace_minj,
                    ]
                )
                - 0.5
            )
            replace_yy = (
                np.array(
                    [
                        replace_mini,
                        replace_mini,
                        replace_maxi,
                        replace_maxi,
                        replace_mini,
                    ]
                )
                - 0.5
            )
            ax2.plot(replace_xx, replace_yy, "-or")

            copy_xx = (
                np.array([copy_minj, copy_maxj, copy_maxj, copy_minj, copy_minj])
                - 0.5
                - crop_minj
            )
            copy_yy = (
                np.array([copy_mini, copy_mini, copy_maxi, copy_maxi, copy_mini])
                - 0.5
                - crop_mini
            )
            ax2.plot(copy_xx, copy_yy, "-ob")

            # zoom in on the area of interest
            # mini, minj, maxi, maxj = params['bbox']
            # ax1.set_xlim(minj - 10, maxj + 10)
            # ax1.set_ylim(mini - 10, maxi + 10)
            # ax2.set_xlim(minj - 10, maxj + 10)
            # ax2.set_ylim(mini - 10, maxi + 10)

            ax1.set_axis_off()
            ax2.set_axis_off()

            plt.tight_layout()
            plt.show()
            plt.close()

    else:
        new_segmentation = np.copy(segmentation)
        new_segmentation[
            :, replace_mini:replace_maxi, replace_minj:replace_maxj
        ] = copied_seg

        # visualize the action
        if False:
            fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

            ax1.imshow(segmentation[0], cmap=random_cmap)

            replace_xx = (
                np.array(
                    [
                        replace_minj,
                        replace_maxj,
                        replace_maxj,
                        replace_minj,
                        replace_minj,
                    ]
                )
                - 0.5
            )
            replace_yy = (
                np.array(
                    [
                        replace_mini,
                        replace_mini,
                        replace_maxi,
                        replace_maxi,
                        replace_mini,
                    ]
                )
                - 0.5
            )
            ax1.plot(replace_xx, replace_yy, "-or")

            copy_xx = (
                np.array([copy_minj, copy_maxj, copy_maxj, copy_minj, copy_minj]) - 0.5
            )
            copy_yy = (
                np.array([copy_mini, copy_mini, copy_maxi, copy_maxi, copy_mini]) - 0.5
            )
            ax1.plot(copy_xx, copy_yy, "-ob")

            # visualize the segmentation after the shift
            ax2.imshow(new_segmentation[0], cmap=random_cmap)

            replace_xx = (
                np.array(
                    [
                        replace_minj,
                        replace_maxj,
                        replace_maxj,
                        replace_minj,
                        replace_minj,
                    ]
                )
                - 0.5
            )
            replace_yy = (
                np.array(
                    [
                        replace_mini,
                        replace_mini,
                        replace_maxi,
                        replace_maxi,
                        replace_mini,
                    ]
                )
                - 0.5
            )
            ax2.plot(replace_xx, replace_yy, "-or")

            copy_xx = (
                np.array([copy_minj, copy_maxj, copy_maxj, copy_minj, copy_minj]) - 0.5
            )
            copy_yy = (
                np.array([copy_mini, copy_mini, copy_maxi, copy_maxi, copy_mini]) - 0.5
            )
            ax2.plot(copy_xx, copy_yy, "-ob")

            # zoom in on the area of interest
            # mini, minj, maxi, maxj = params['bbox']
            # ax1.set_xlim(minj - 10, maxj + 10)
            # ax1.set_ylim(mini - 10, maxi + 10)
            # ax2.set_xlim(minj - 10, maxj + 10)
            # ax2.set_ylim(mini - 10, maxi + 10)

            ax1.set_axis_off()
            ax2.set_axis_off()

            plt.tight_layout()
            plt.show()
            plt.close()

    return new_segmentation


def perform_action(segmentation, params, hparams):
    # call the correct action function
    action_dict = {
        "edge_shift": shift_edge,
        "merge_regions": merge_regions,
    }

    new_segmentation = action_dict[params["action"]](
        segmentation, params, hparams, vis=False
    )

    return new_segmentation


def get_graph_edges(ins_crop):
    nodes, edges = parallel.get_graph(ins_crop, small_crop=True)
    h, w = ins_crop.shape
    G = nx.Graph()

    for (a, b) in edges:
        ai, aj = int(a[0]), int(a[1])
        bi, bj = int(b[0]), int(b[1])

        # this is a horizontal edge
        if ai == bi:
            n_mini = max(ai - 1, 0)
            n_maxi = min(ai + 1, h)
            n_minj = max(min(aj, bj), 0)
            n_maxj = min(max(aj, bj), w)

        # this is a vertical edge
        elif aj == bj:
            n_mini = max(min(ai, bi), 0)
            n_maxi = min(max(ai, bi), h)
            n_minj = max(aj - 1, 0)
            n_maxj = min(aj + 1, w)

        else:
            raise Exception("non-manhattan edge")

        neighbors = np.unique(ins_crop[n_mini:n_maxi, n_minj:n_maxj])

        # normal boundary
        if len(neighbors) == 2:
            (a, b) = neighbors.tolist()
            G.add_edge(a, b)

        # boundary along edge of image
        elif len(neighbors) == 1:
            a = neighbors.tolist()[0]
            if not G.has_node(a):
                G.add_node(a)

        else:
            raise Exception("cannot have more than 2 neighbors!")

    graph_edges = []

    for ins_a in G.nodes:
        for ins_b in G.nodes:
            if ins_a > ins_b:
                if G.has_edge(ins_a, ins_b):
                    graph_edges.append([ins_a, 1, ins_b])
                else:
                    graph_edges.append([ins_a, -1, ins_b])

    return graph_edges


def vis_pair(old_seg, new_seg, params):
    (a, b) = params["line"][0], params["line"][-1]
    center_i = int((a[0] + b[0]) / 2)
    center_j = int((a[1] + b[1]) / 2)

    side_len = 128
    mini = max(center_i - side_len // 2, 0)
    minj = max(center_j - side_len // 2, 0)
    mini = min(mini, old_seg[0].shape[0] - side_len)
    minj = min(minj, old_seg[0].shape[1] - side_len)
    maxi = min(mini + side_len, old_seg[0].shape[0])
    maxj = min(minj + side_len, old_seg[0].shape[1])

    fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

    ax1.imshow(
        old_seg[1][mini:maxi, minj:maxj] / 7.0,
        cmap=my_cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax2.imshow(
        new_seg[1][mini:maxi, minj:maxj] / 7.0,
        cmap=my_cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    ax1.set_axis_off()
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()
    plt.close()


def action_ok(old_seg, new_seg, params):
    # make sure that no instances are gone or no new instances
    tmp_seg = measure.label(new_seg[0], background=-1, connectivity=1)
    if (len(np.unique(old_seg[0])) != len(np.unique(tmp_seg))) or (
        len(np.unique(old_seg[0])) != len(np.unique(new_seg[0]))
    ):
        return False

    # finally check that graph is not being edited
    mini, minj, maxi, maxj = params["bbox"]
    mini = max(mini - 16, 0)
    minj = max(minj - 16, 0)
    maxi = min(maxi + 16, old_seg[0].shape[0])
    maxj = min(maxj + 16, old_seg[0].shape[1])

    old_ins_crop = old_seg[0][mini:maxi, minj:maxj]
    new_ins_crop = new_seg[0][mini:maxi, minj:maxj]

    old_edges = sorted(get_graph_edges(old_ins_crop))
    new_edges = sorted(get_graph_edges(new_ins_crop))

    if old_edges != new_edges:
        return False

    # then check that we are not shifting out of the original bounds
    line = np.array(params["line"]) + 1
    (a, b) = line[0], line[-1]

    if params["orientation"] == "H":
        assert a[0] == b[0]
        mini = a[0] - 1
        maxi = a[0] + 1
        minj = min(a[1], b[1])
        maxj = max(a[1], b[1])

    elif params["orientation"] == "V":
        assert a[1] == b[1]
        mini = min(a[0], b[0])
        maxi = max(a[0], b[0])
        minj = a[1] - 1
        maxj = a[1] + 1

    for affected_id in np.unique(old_seg[0, mini:maxi, minj:maxj]):
        invalid_area = ~binary_dilation(old_seg[0] == affected_id, square(7))

        if (new_seg[0] == affected_id)[invalid_area].any():
            return False

    return True


def take_heuristic_actions(graph, segmentation):
    hparams = {"num_threads": 1, "short_edge_threshold": 2}

    actions_taken = []
    while True:
        all_actions = get_all_edge_collapses(graph, segmentation, hparams)

        if not len(all_actions):
            break

        affected_mask = np.zeros_like(segmentation[0], dtype=np.uint8)
        for params in all_actions:
            if params in actions_taken:
                continue

            # something else has been done in this area, so ignore this action
            mini, minj, maxi, maxj = params["bbox"]
            if affected_mask[mini - 1 : maxi + 1, minj - 1 : maxj + 1].sum():
                continue

            # since we do a lot of actions in a row, we don't do r2v after each
            # step, but once after
            new_segmentation = perform_action(segmentation, params, hparams)
            actions_taken.append(params)

            if action_ok(segmentation, new_segmentation, params):
                segmentation = new_segmentation
            else:
                # print(params['bbox'])
                # fig, [ax1, ax2] = plt.subplots(ncols=2)
                # ax1.imshow(segmentation[1], cmap='nipy_spectral')
                # ax2.imshow(new_segmentation[1], cmap='nipy_spectral')

                # yy = [mini-3, mini-3, maxi+3, maxi+3, mini-3]
                # xx = [minj-3, maxj+3, maxj+3, minj-3, minj-3]
                # plt.plot(xx, yy, '-r')

                # plt.show()
                # plt.close()
                continue

            # update our affected mask with region just now affected
            affected_mask[mini:maxi, minj:maxj] = 1

        if not affected_mask.sum():
            break

        # now that segmentation is cleaned up, get the graph of this new one
        graph, _ = parallel.get_graph(segmentation[0], multi=False)

    return graph, segmentation


def do_heuristic(fp_id):
    print(fp_id)

    # if os.path.exists(utils.PREPROCESS_ROOT + 'heuristic/%s.npy' % fp_id):
    #   return

    # note that "segmentation" is instance and semantic masks stacked together
    sem_full = np.load(utils.PREPROCESS_ROOT + "semantic_pred/%s.npy" % fp_id)
    ins_full = measure.label(sem_full, background=-1, connectivity=1)
    segmentation = np.stack([ins_full, sem_full], axis=0)
    graph, _ = parallel.get_graph(segmentation[0], multi=False)
    _, new_segmentation = take_heuristic_actions(graph, segmentation)

    np.save(
        utils.PREPROCESS_ROOT + "heuristic/%s.npy" % fp_id,
        new_segmentation[1],
        allow_pickle=False,
    )

    old_img = my_cmap(segmentation[1] / 7.0)[:, :, :3] * 255.0
    new_img = my_cmap(new_segmentation[1] / 7.0)[:, :, :3] * 255.0
    sem_img = np.concatenate([old_img, new_img], axis=1)
    sem_img = Image.fromarray(sem_img.astype("uint8"))
    sem_img.save(utils.PREPROCESS_ROOT + "heuristic/%s.png" % fp_id)


def do_refined_heuristic(fp_id):
    print(fp_id)

    # note that "segmentation" is instance and semantic masks stacked together
    sem_full = np.load(utils.PREPROCESS_ROOT + "refined/%s.npy" % fp_id)
    ins_full = measure.label(sem_full, background=-1, connectivity=1)
    segmentation = np.stack([ins_full, sem_full], axis=0)
    graph, _ = parallel.get_graph(segmentation[0], multi=False)
    _, new_segmentation = take_heuristic_actions(graph, segmentation)

    np.save(
        utils.PREPROCESS_ROOT + "refined_heuristic/%s.npy" % fp_id,
        new_segmentation[1],
        allow_pickle=False,
    )

    old_img = my_cmap(segmentation[1] / 7.0)[:, :, :3] * 255.0
    new_img = my_cmap(new_segmentation[1] / 7.0)[:, :, :3] * 255.0
    sem_img = np.concatenate([old_img, new_img], axis=1)
    sem_img = Image.fromarray(sem_img.astype("uint8"))
    sem_img.save(utils.PREPROCESS_ROOT + "refined_heuristic/%s.png" % fp_id)


def do_heuristic_refined_heuristic(fp_id):
    print(fp_id)

    # note that "segmentation" is instance and semantic masks stacked together
    sem_full = np.load(utils.PREPROCESS_ROOT + "heuristic_refined/%s.npy" % fp_id)
    ins_full = measure.label(sem_full, background=-1, connectivity=1)
    segmentation = np.stack([ins_full, sem_full], axis=0)
    graph, _ = parallel.get_graph(segmentation[0], multi=False)
    _, new_segmentation = take_heuristic_actions(graph, segmentation)

    np.save(
        utils.PREPROCESS_ROOT + "heuristic_refined_heuristic/%s.npy" % fp_id,
        new_segmentation[1],
        allow_pickle=False,
    )

    old_img = my_cmap(segmentation[1] / 7.0)[:, :, :3] * 255.0
    new_img = my_cmap(new_segmentation[1] / 7.0)[:, :, :3] * 255.0
    sem_img = np.concatenate([old_img, new_img], axis=1)
    sem_img = Image.fromarray(sem_img.astype("uint8"))
    sem_img.save(utils.PREPROCESS_ROOT + "heuristic_refined_heuristic/%s.png" % fp_id)


if __name__ == "__main__":
    args = utils.parse_arguments()

    semantic_fs = glob.glob(utils.PREPROCESS_ROOT + "instance_pred/*.npy")
    fp_ids = [x.split("/")[-1].split(".")[0] for x in semantic_fs]

    if args.method == "h":
        print("Heuristic")
        os.makedirs(utils.PREPROCESS_ROOT + "heuristic/", exist_ok=True)

        # for fp_id in fp_ids:
        #   do_heuristic(fp_id)

        with Pool(args.num_threads) as p:
            p.map(do_heuristic, fp_ids)

    elif args.method == "rh":
        print("Refined -> heuristic")
        os.makedirs(utils.PREPROCESS_ROOT + "refined_heuristic/", exist_ok=True)

        with Pool(args.num_threads) as p:
            p.map(do_refined_heuristic, fp_ids)

    elif args.method == "hrh":
        print("Heuristic -> refined -> heuristic")
        os.makedirs(
            utils.PREPROCESS_ROOT + "heuristic_refined_heuristic/", exist_ok=True
        )

        with Pool(args.num_threads) as p:
            p.map(do_heuristic_refined_heuristic, fp_ids)

    else:
        raise Exception("Unknown method")
