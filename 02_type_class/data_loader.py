from __future__ import print_function, division

import os
import json
import glob

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from tqdm import tqdm
from skimage import measure
from skimage.transform import resize, rotate
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

import utils
import paths


# bg is ignored as a class here, as that's handled by heuristics
class_names = ["outer", "inner", "window", "door", "frame", "room", "symbol"]


class FloorplanSegmentation(Dataset):
    def __init__(self, split_ids):
        super().__init__()

        fp_ids = []

        for split_id in split_ids:
            split_json = paths.SPLITS_ROOT + "ids_%d.json" % split_id
            with open(split_json, "r") as f:
                fp_ids.extend(json.load(f))

        self.actual_len = len(fp_ids)

        # load the data into memory
        self.data = {}
        self.examples = []
        self.all_labels = []

        print(">>> Caching %d floorplans..." % len(fp_ids))
        for fp_id in tqdm(fp_ids):
            # load the floorplan image and label
            image = Image.open(paths.IMG_ROOT + "%s.jpg" % fp_id)
            image = np.array(image, dtype=np.float32) / 255.0

            semantic_gt = np.load(paths.GT_SEMANTIC_ROOT + "%s.npy" % fp_id)
            instance_gt = np.load(paths.GT_INSTANCE_ROOT + "%s.npy" % fp_id)
            semantic_vote = np.load(paths.VOTE_SEMANTIC_ROOT + "%s.npy" % fp_id)
            instance_pred = np.load(paths.PRED_INSTANCE_ROOT + "%s.npy" % fp_id)

            assert image.shape == semantic_gt.shape == semantic_vote.shape

            # 24 is the door/window symbol, shift it down to 7
            semantic_gt[semantic_gt == 24] = 7
            semantic_vote[semantic_vote == 24] = 7

            # cache the data
            self.data[fp_id] = (
                image,
                semantic_gt,
                instance_gt,
                semantic_vote,
                instance_pred,
            )

            # save all possible samples
            self.cache_examples(semantic_gt, instance_gt, fp_id, "gt")
            self.cache_examples(semantic_vote, instance_pred, fp_id, "pred")

        # it can't hurt to shuffle here as well...
        np.random.shuffle(self.examples)

    def cache_examples(self, semantic_full, instance_full, fp_id, name):
        for instance_id in np.unique(instance_full):
            instance_mask = instance_full == instance_id
            instance_sem = np.unique(semantic_full[instance_mask])
            assert len(instance_sem) == 1
            instance_sem = instance_sem[0] - 1  # bg is not a class here

            # skip background
            if instance_sem == -1:
                continue

            # sample seed points
            else:
                ii, jj = np.nonzero(instance_mask)
                seeds = np.array(list(zip(ii, jj)))

                seed_idx = np.random.choice(
                    range(len(seeds)), size=min(20, len(seeds)), replace=False
                )
                seeds = seeds[seed_idx]

                side_len = 224
                for center_i, center_j in seeds:
                    mini = max(center_i - side_len // 2, 0)
                    minj = max(center_j - side_len // 2, 0)
                    mini = min(mini, instance_full.shape[0] - side_len)
                    minj = min(minj, instance_full.shape[1] - side_len)
                    maxi = min(mini + side_len, instance_full.shape[0])
                    maxj = min(minj + side_len, instance_full.shape[1])

                    assert (mini >= 0) and (minj >= 0)
                    assert (maxi <= instance_full.shape[0]) and (
                        maxj <= instance_full.shape[1]
                    )
                    assert ((maxi - mini) == side_len) and ((maxj - minj) == side_len)

                    bbox = [mini, minj, maxi, maxj]
                    self.examples.append((fp_id, name, instance_id, instance_sem, bbox))
                    self.all_labels.append(instance_sem)

    def nn_resize(self, image, size):
        return resize(image, size, order=0, preserve_range=True, anti_aliasing=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        fp_id, name, instance_id, instance_sem, bbox = self.examples[index]

        image, semantic_gt, instance_gt, semantic_vote, instance_pred = self.data[fp_id]

        # make a new copy to make sure we're not modifying the original
        mini, minj, maxi, maxj = bbox
        image_crop = np.copy(image[mini:maxi, minj:maxj])

        if name == "pred":
            instance_crop = np.copy(instance_pred[mini:maxi, minj:maxj])
        elif name == "gt":
            instance_crop = np.copy(instance_gt[mini:maxi, minj:maxj])
        else:
            raise Exception

        # normalize this crop of image
        # if image_crop.max() > 0.0:
        #   image_crop = image_crop / image_crop.max()

        # highlight a particular instance
        instance_mask = (instance_crop == instance_id).astype(np.float32)

        # resize into network inputs
        # input_size = [224, 224]
        # image_crop = resize(image_crop, input_size)
        # instance_mask = self.nn_resize(instance_mask, input_size)

        # convert things to PyTorch tensors
        image_crop = torch.Tensor(image_crop)
        instance_mask = torch.Tensor(instance_mask)

        combined = torch.stack([image_crop, instance_mask], axis=0)

        # bg is not a class here
        assert instance_sem >= 0 and instance_sem <= 6
        return combined, instance_sem


if __name__ == "__main__":
    import utils
    import matplotlib.pyplot as plt

    fp_dataset = FloorplanSegmentation(
        [
            0,
        ]
    )

    for idx, (combined, label) in enumerate(fp_dataset):
        if idx == 1000:
            break

        plt.imshow(combined[0], cmap="gray")
        plt.imshow(combined[1], cmap="hot", alpha=0.7)
        plt.axis("off")
        plt.title(class_names[label])
        plt.tight_layout()
        # plt.show()
        plt.savefig("temp/%06d.png" % idx)
        plt.close()
