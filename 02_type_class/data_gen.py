import glob
import os
import shutil
from multiprocessing import Pool
from os.path import join as pj

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paths
import process_boundary
import utils
from PIL import Image

args = utils.parse_arguments()

nipy_cmap = plt.get_cmap("nipy_spectral")
random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))


def job(fp_id):
    print(fp_id)

    # save predicted instance
    instance_pred = process_boundary.load_and_segment(fp_id)
    np.save(
        paths.PRED_INSTANCE_ROOT + "%s.npy" % fp_id, instance_pred, allow_pickle=False
    )

    if not os.path.exists(paths.GT_SEMANTIC_ROOT):
        return

    # vote on semantics of predicted instance
    semantic_gt = np.load(paths.GT_SEMANTIC_ROOT + "%s.npy" % fp_id)
    semantic_vote = np.zeros_like(semantic_gt)

    for instance_id in np.unique(instance_pred):
        instance_mask = instance_pred == instance_id

        # vote on the instance label
        labels, counts = np.unique(semantic_gt[instance_mask], return_counts=True)
        instance_label = labels[np.argmax(counts)]

        # NOTE sometimes doors are predicted thicker, and room gets selected
        # So we have a hack here to recover those doors
        if (
            (instance_label == 6)
            and (max(counts) / np.sum(counts) < 0.7)
            and len(counts) > 1
        ):
            instance_label = labels[np.argsort(counts)[-2]]

        # assign this label to our placeholder mask
        semantic_vote[instance_mask] = instance_label

    np.save(paths.VOTE_SEMANTIC_ROOT + "%s.npy" % fp_id, semantic_vote)

    # save three-way crop visualization of instance, semantic, ang GT semantic
    instance_img = random_cmap(instance_pred)
    sem_vote_img = nipy_cmap(semantic_vote / 7.0)
    sem_gt_img = nipy_cmap(semantic_gt / 7.0)

    # pad to multiples of crop size
    crop_size = 512
    crop_stride = 256

    height, width, channels = instance_img.shape

    pad_height = (height // crop_size + 1) * crop_size
    pad_width = (width // crop_size + 1) * crop_size

    pad_instance = np.zeros([pad_height, pad_width, channels], dtype=instance_img.dtype)
    pad_instance[:height, :width] = instance_img

    pad_sem_vote = np.zeros([pad_height, pad_width, channels], dtype=sem_vote_img.dtype)
    pad_sem_vote[:height, :width] = sem_vote_img

    pad_sem_gt = np.zeros([pad_height, pad_width, channels], dtype=sem_gt_img.dtype)
    pad_sem_gt[:height, :width] = sem_gt_img

    # densely crop
    crop_i = 0  # used to save the crop json
    pad_height, pad_width, _ = pad_instance.shape

    for mini in range(0, pad_height - crop_size + 1, crop_stride):
        for minj in range(0, pad_width - crop_size + 1, crop_stride):
            maxi = mini + crop_size
            maxj = minj + crop_size

            # make sure bbox is within bounds
            assert maxi <= pad_height and maxj <= pad_width

            instance_crop = pad_instance[mini:maxi, minj:maxj]
            sem_vote_crop = pad_sem_vote[mini:maxi, minj:maxj]
            sem_gt_crop = pad_sem_gt[mini:maxi, minj:maxj]

            combined_crop = np.concatenate(
                [instance_crop, sem_vote_crop, sem_gt_crop], axis=1
            )
            combined_img = Image.fromarray(np.uint8(combined_crop * 255.0), mode="RGBA")
            combined_img.save(
                paths.GA_ROOT
                + "preprocess/visualize_vote/%s_%02d.png" % (fp_id, crop_i)
            )

            crop_i += 1


if __name__ == "__main__":
    if args.restart and os.path.exists(paths.PRED_INSTANCE_ROOT):
        shutil.rmtree(paths.PRED_INSTANCE_ROOT)
        shutil.rmtree(paths.VOTE_SEMANTIC_ROOT)
        shutil.rmtree(paths.GA_ROOT + "preprocess/visualize_vote/")

    os.makedirs(paths.PRED_INSTANCE_ROOT, exist_ok=True)
    os.makedirs(paths.VOTE_SEMANTIC_ROOT, exist_ok=True)
    os.makedirs(paths.GA_ROOT + "preprocess/visualize_vote/", exist_ok=True)

    if args.test_folder:
        print('User-specified test folder')
        img_files = glob.glob(pj(args.test_folder, "*"))
        fp_ids = [x.split("/")[-1].split(".")[0] for x in img_files]
    else:
        semantic_files = glob.glob(pj(paths.IMG_ROOT, "*"))
        fp_ids = [x.split("/")[-1].split(".")[0] for x in semantic_files]

    with Pool(5) as p:
        p.map(job, fp_ids)
