import os
import glob
import json
import copy
import shutil

from multiprocessing import Pool

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from skimage import measure
from skimage.draw import polygon, polygon_perimeter
from skimage.morphology import erosion, dilation, square
from matplotlib.colors import ListedColormap
from shapely.geometry import Polygon, Point, box

import utils
import mask_fixer


# some paths
GA_ROOT = "../data/"
PREPROCESS_ROOT = "../data/preprocess/"
ANNO_ROOT = "../data/"
IMG_ROOT = "../data/raw_img/"


args = utils.parse_arguments()
class_map = {
    "bg": 0,
    "outer": 1,
    "inner": 2,
    "window": 3,
    "door": 4,
    "frame": 5,
    "room": 6,
    "symbol": 7,
}
reverse_class_map = dict(zip(class_map.values(), class_map.keys()))


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


def plot_poly(poly, ori_poly):
    fig, [ax1, ax2] = plt.subplots(ncols=2)

    xx, yy = poly.exterior.coords.xy
    ax1.plot(xx, yy, "-o")

    xx, yy = ori_poly.exterior.coords.xy
    ax2.plot(xx, yy, "-o")

    plt.gca().axis("equal")
    plt.show()
    plt.close()


def compute_angle(p1, p2, p3):
    a = np.array(p2) - np.array(p1)
    b = np.array(p2) - np.array(p3)

    a_norm = np.sqrt(np.sum(a**2, axis=-1))
    b_norm = np.sqrt(np.sum(b**2, axis=-1))

    return np.rad2deg(np.arccos(np.sum(a * b, axis=-1) / (a_norm * b_norm)))


# remove pairwise duplicates, if there are any
def remove_duplicates(points):
    unique_points = []

    prev = None

    for point in points:
        # first point, so add it
        if prev == None:
            unique_points.append(point)
            prev = point

        # this new point is different from last, so add it
        elif point != prev:
            unique_points.append(point)
            prev = point

        # the same point as last, ignore it
        else:
            continue

    # sometimes first and last points are the same, remove the last one
    if unique_points[0] == unique_points[-1]:
        unique_points = unique_points[:-1]

    return unique_points


def remove_collinear(points, angle_threshold=5):
    # make sure we always have triplets of points to look at
    points_loop = copy.deepcopy(points)
    points_loop.insert(0, points_loop[-1])
    points_loop.append(points_loop[1])

    # look at the angle formed by three consecutive points, and if it's close
    # enough to 180, then it's collinear and we discard the middle point
    collinear_points = []

    for p1, p2, p3 in zip(points_loop[:-2], points_loop[1:-1], points_loop[2:]):
        if abs(compute_angle(p1, p2, p3) - 180) >= angle_threshold:
            collinear_points.append(p2)

    return collinear_points


# check if a line is APPROXIMATELY manhattan
# p1 is the start, p2 is the end, p3 is a point axis-aligned with p1
def line_is_manhattan(p1, p2, angle_threshold=5):
    x1, y1 = p1
    x2, y2 = p2

    # if the angle formed by the line and one of the axis is more than the
    # threshold, then this line we do not try to fix
    if abs(x1 - x2) > abs(y1 - y2):  # horizontal line
        x3 = x2
        y3 = y1
    elif abs(x1 - x2) < abs(y1 - y2):  # vertical line
        x3 = x1
        y3 = y2
    else:
        # this is a diagnoal line, so it doesn't really matter
        x3 = x1
        y3 = y2

    # if the angle formed by the line and one of the axis is more than the
    # threshold, then this line is not manhattan, and we do not try to fix
    if compute_angle((x2, y2), (x1, y1), (x3, y3)) > angle_threshold:
        return False
    else:
        return True


# shift points so they follow manhattan mostly
def ensure_manhattan(points, angle_threshold=5):
    # check lines formed by pairs of points
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        # only do things if line is not yet manhattan
        if (x1 != x2) and (y1 != y2):
            if not line_is_manhattan(points[i], points[i + 1]):
                continue

            # fixed line based on if it's a horizontal or vertical line
            if abs(x1 - x2) > abs(y1 - y2):  # horizontal line (y needs to be same)
                y1 = y2 = (y1 + y2) // 2
            elif abs(x1 - x2) < abs(y1 - y2):  # vertical line (x needs to be same)
                x1 = x2 = (x1 + x2) // 2
            else:
                raise Exception("Bad line annotation, manually fix")

            # make sure we didn't break the manhattan rule of the previous line
            if i >= 1 and line_is_manhattan(points[i - 1], points[i]):
                x0, y0 = points[i - 1]
                assert x0 == x1 or y0 == y1

            # check passed, we can modify points
            points[i] = (x1, y1)
            points[i + 1] = (x2, y2)

    # last point is special case, make it manhattan to both the first point and
    # second to last point if possible
    if line_is_manhattan(points[-1], points[0]):
        x1, y1 = points[-2]
        x2, y2 = points[-1]
        x3, y3 = points[0]

        if x1 == x2:
            y2 = y3
        elif y1 == y2:
            x2 = x3
        else:
            # determine if it's a horizontal or vertical line
            if abs(x3 - x2) > abs(y3 - y2):  # horizontal line
                y2 = y3
            elif abs(x3 - x2) < abs(y3 - y2):  # vertical line
                x2 = x3
            else:
                raise Exception("Bad line annotation, manually fix")

        points[-1] = (x2, y2)

    # double check to make sure points are manhattan
    for i in range(len(points) - 1):
        if line_is_manhattan(points[i], points[i + 1]):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            assert x1 == x2 or y1 == y2

    return points


def to_poly(region):
    if region["shape_attributes"]["name"] == "polygon":
        ori_points = list(
            zip(
                region["shape_attributes"]["all_points_x"],
                region["shape_attributes"]["all_points_y"],
            )
        )
        points = remove_duplicates(ori_points)
        points = remove_collinear(points)
        # points = ensure_manhattan(points)
        poly = Polygon(points)

        return poly, Polygon(ori_points)

    elif region["shape_attributes"]["name"] == "rect":
        minx = region["shape_attributes"]["x"]
        miny = region["shape_attributes"]["y"]
        maxx = minx + region["shape_attributes"]["width"]
        maxy = miny + region["shape_attributes"]["height"]

        poly = box(minx, miny, maxx, maxy)
        assert poly.is_valid

        return poly, poly

    else:
        raise Exception("Unknown shape name ", region["shape_attributes"]["name"])


def preprocess_floorplan(fp_anno):
    floorplan_id = fp_anno["filename"].strip().split(".jpg")[0]

    # load the floorplan image
    img_path = IMG_ROOT + floorplan_id + ".jpg"
    fp_img = Image.open(img_path).convert("L")
    fp_img = np.array(fp_img, dtype=np.float32) / 255.0
    height, width = fp_img.shape

    # empty mask placeholders
    semantic_mask = np.zeros([height, width], dtype=int)
    instance_mask = np.zeros([height, width], dtype=int)
    boundary_mask = np.zeros([height, width], dtype=int)

    # paste the wall and opening annotations first
    for region_idx, region in enumerate(fp_anno["regions"]):
        if "label" not in region["region_attributes"].keys():
            print(floorplan_id)
            continue

        label = region["region_attributes"]["label"]

        if label == "sdf":
            continue

        poly, ori_poly = to_poly(region)

        if not poly.is_valid:
            print(floorplan_id, region_idx)
            if True:
                plot_poly(poly, ori_poly)

            continue

        xx, yy = poly.exterior.coords.xy
        rr, cc = polygon(yy, xx, shape=[height, width])

        # wall Annotations
        if label in ["outer", "inner"]:
            attribute = region["region_attributes"]["attribute"]
            wall_id, mask_type = attribute.split("_")

            if mask_type == "pos":
                semantic_mask[rr, cc] = class_map[label]
                instance_mask[rr, cc] = instance_mask.max() + 1

            elif mask_type == "neg":
                semantic_mask[rr, cc] = class_map["bg"]
                instance_mask[rr, cc] = 0

            else:
                raise Exception("Unknown wall mask type")

        # other things annotation
        # some rooms are really small, so they are manually annotated
        elif label in ["window", "door", "portal", "room", "frame"]:
            semantic_mask[rr, cc] = class_map[label]
            instance_mask[rr, cc] = instance_mask.max() + 1

        else:
            print("Unknown key %s in %s" % (label, floorplan_id))

        # regardless of what it is, mark its boundary
        rr_b, cc_b = polygon_perimeter(yy, xx, shape=[height, width])
        boundary_mask[rr_b, cc_b] = 1

    """
    Room Annotation

    For this, we need to combine all the wall and opening masks together,
    then we fill in some small gaps and find connected components.
  """
    negative_mask = semantic_mask.copy()
    negative_mask[negative_mask > 0] = 1

    # fill in gaps between walls and openings
    k = 3
    negative_mask = dilation(negative_mask, square(k))

    # flip the negative into positive mask
    positive_mask = np.logical_not(negative_mask).astype(int)

    # find connected components (note we dilate the region masks)
    all_labels = measure.label(positive_mask, background=0)
    all_labels = dilation(all_labels, square(k))

    # set wall and background pixels to bg class, and other pixels to room
    room_semantic_mask = all_labels.copy()
    room_semantic_mask[room_semantic_mask == 1] = 0
    room_semantic_mask[room_semantic_mask > 1] = class_map["room"]
    semantic_mask += room_semantic_mask
    # assert (semantic_mask <= class_map['room']).all()

    # get room instances mask
    room_instance_mask = all_labels.copy()
    room_instance_mask = np.maximum(0, room_instance_mask - 1)
    room_instance_mask[room_instance_mask != 0] += instance_mask.max()
    instance_mask += room_instance_mask
    assert (instance_mask <= instance_mask.max()).all()

    # generate the instance boundary map in region format
    # boundary_mask = utils.generate_boundary_mask(instance_mask)

    # remove the black bits in the two masks, see comment on top of
    # mask_fixer.py for details
    # semantic_mask = mask_fixer.remove_black_bits(fp_img, semantic_mask)
    # instance_mask = mask_fixer.remove_black_bits(fp_img, instance_mask)
    semantic_mask = mask_fixer.fill_long_gaps(floorplan_id, fp_img, semantic_mask)
    instance_mask = mask_fixer.fill_long_gaps(floorplan_id, fp_img, instance_mask)

    # NOTE we really only care about separate semantic instances, so our
    # instances really should come from components in semantic masks instead
    instance_mask = measure.label(semantic_mask, background=0, connectivity=1)

    for ins_id in np.unique(instance_mask):
        ins_mask = instance_mask == ins_id

        if ins_mask.sum() == 1:
            ii, jj = np.nonzero(ins_mask)
            assert len(ii) == 1 and len(jj) == 1

            mini = ii[0] - 1
            maxi = ii[0] + 2
            minj = jj[0] - 1
            maxj = jj[0] + 2
            ins_mask[mini:maxi, minj:maxj] = True

            unique, counts = np.unique(semantic_mask[ins_mask], return_counts=True)
            new_sem = unique[np.argmax(counts)]
            semantic_mask[ii[0], jj[0]] = new_sem

    instance_mask = measure.label(semantic_mask, background=0, connectivity=1)

    """
    Crop and reshape the image and all the masks to a bbox
  """
    bbox = utils.get_floorplan_bbox(semantic_mask, instance_mask)
    mini, minj, maxi, maxj = bbox

    fp_img = fp_img[mini:maxi, minj:maxj]
    boundary_mask = boundary_mask[mini:maxi, minj:maxj]
    semantic_mask = semantic_mask[mini:maxi, minj:maxj]
    instance_mask = instance_mask[mini:maxi, minj:maxj]

    """
    Save all the stuff
  """
    fp_img_path = PREPROCESS_ROOT + "fp_img/%s.jpg" % floorplan_id
    semantic_path = PREPROCESS_ROOT + "semantic/%s.npy" % floorplan_id
    instance_path = PREPROCESS_ROOT + "instance/%s.npy" % floorplan_id
    bbox_path = PREPROCESS_ROOT + "bbox/%s.csv" % floorplan_id
    boundary_path = PREPROCESS_ROOT + "boundary/%s.npy" % floorplan_id

    utils.ensure_dir(fp_img_path)
    utils.ensure_dir(semantic_path)
    utils.ensure_dir(instance_path)
    utils.ensure_dir(bbox_path)
    utils.ensure_dir(boundary_path)

    fp_img = Image.fromarray(np.uint8(fp_img * 255), "L")
    fp_img.save(fp_img_path)
    np.save(semantic_path, semantic_mask, allow_pickle=False)
    np.save(instance_path, instance_mask, allow_pickle=False)
    np.save(boundary_path, boundary_mask, allow_pickle=False)

    with open(bbox_path, "w") as f:
        f.write("mini,minj,maxi,maxj\n")
        f.write(",".join([str(x) for x in bbox]))

    """
    Visualization
  """
    # save full semantic and instance visualization
    rand_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))

    semantic_image = my_cmap(semantic_mask / semantic_mask.max())
    instance_image = rand_cmap(instance_mask / instance_mask.max())

    semantic_image = Image.fromarray(np.uint8(semantic_image * 255.0), mode="RGBA")
    instance_image = Image.fromarray(np.uint8(instance_image * 255.0), mode="RGBA")

    # semantic_image = Image.blend(fp_img.convert('RGBA'), semantic_image, alpha=0.8)

    semantic_path = PREPROCESS_ROOT + "visualize_full/%s_sem.png" % floorplan_id
    instance_path = PREPROCESS_ROOT + "visualize_full/%s_ins.png" % floorplan_id

    utils.ensure_dir(semantic_path)
    semantic_image.save(semantic_path)
    instance_image.save(instance_path)


if __name__ == "__main__":
    if args.restart:
        if os.path.exists(PREPROCESS_ROOT):
            shutil.rmtree(PREPROCESS_ROOT)

    if os.path.exists(PREPROCESS_ROOT):
        raise Exception("Data already preprocessed! Use --restart")

    # load the annotations
    with open(ANNO_ROOT + "via_annotations.json", "r") as f:
        all_anno = json.load(f)

    print("Processing %d floorplans" % len(all_anno.keys()))

    if args.single:
        # single-threaded
        for fp_anno in tqdm(all_anno.values()):
            preprocess_floorplan(fp_anno)

    else:
        # multi-threaded
        jobs = list(all_anno.values())
        with Pool(5) as p:
            p.map(preprocess_floorplan, jobs)
