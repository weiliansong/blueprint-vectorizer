import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import measure
from torch.utils.data import Dataset
from tqdm import tqdm

import data_utils
import find_corr
import paths
import sem_rings
import utils

interp_mode = transforms.InterpolationMode


class FloorplanDataset(Dataset):
    def __init__(self, split_ids, hparams, cnn_only=False):
        super().__init__()
        self.hparams = hparams
        self.cnn_only = cnn_only
        assert cnn_only
        assert hparams["skip_uncommon"] == False

        # get the corresponding floorplan IDs for the split we're using
        fp_ids = []

        for split_id in split_ids:
            split_json = paths.SPLITS_ROOT + "ids_%d.json" % split_id
            with open(split_json, "r") as f:
                fp_ids.extend(json.load(f))

        # cache the necessary data
        self.fp_img = {}
        self.pred2gt = {}

        self.gt_ins_full = {}
        self.gt_sem_full = {}
        self.gt_ins_sem = {}
        self.gt_hist_dicts = {}
        self.gt_all_sem_rings = {}

        self.pred_ins_full = {}
        self.pred_sem_full = {}
        self.pred_ins_sem = {}
        self.pred_hist_dicts = {}
        self.pred_all_sem_rings = {}

        self.examples = []  # store all the valid doors/windows/frames for training

        print("Caching...")
        pred_skipped = 0
        gt_skipped = 0

        for fp_id in tqdm(fp_ids):
            img = Image.open(paths.IMG_ROOT + "%s.jpg" % fp_id)
            img = np.array(img, dtype=np.float32) / 255.0
            self.fp_img[fp_id] = img

            g_ins_full = np.load(paths.INSTANCE_ROOT + "%s.npy" % fp_id)
            g_sem_full = np.load(paths.SEMANTIC_ROOT + "%s.npy" % fp_id)

            p_ins_full = np.load(paths.PRED_INSTANCE_ROOT + "%s.npy" % fp_id)
            p_sem_full = np.load(paths.PRED_SEMANTIC_ROOT + "%s.npy" % fp_id)

            assert 24 not in g_sem_full
            assert 24 not in p_sem_full

            self.pred2gt[fp_id] = find_corr.find_correspondence(
                p_ins_full, p_sem_full, g_ins_full, g_sem_full
            )

            self.gt_ins_full[fp_id] = g_ins_full
            self.gt_sem_full[fp_id] = g_sem_full
            self.pred_ins_full[fp_id] = p_ins_full
            self.pred_sem_full[fp_id] = p_sem_full

            self.gt_ins_sem[fp_id] = {}
            self.gt_hist_dicts[fp_id] = {}
            self.pred_ins_sem[fp_id] = {}
            self.pred_hist_dicts[fp_id] = {}

            gt_ins_edges = sem_rings.get_instance_edge_mapping(g_ins_full, g_sem_full)
            gt_sem_rings = sem_rings.get_sem_rings(gt_ins_edges, g_ins_full, g_sem_full)

            pred_ins_edges = sem_rings.get_instance_edge_mapping(p_ins_full, p_sem_full)
            pred_sem_rings = sem_rings.get_sem_rings(
                pred_ins_edges, p_ins_full, p_sem_full
            )

            self.gt_all_sem_rings[fp_id] = gt_sem_rings
            self.pred_all_sem_rings[fp_id] = pred_sem_rings

            # this will be used to store all the examples for this floorplan
            fp_examples = set()

            # load GT examples first, so we know frequency information as well
            for gt_id, (gt_sem, gt_ring) in gt_sem_rings.items():
                gt_id = int(gt_id)  # unfortunately JSON only supports str as key...

                self.gt_ins_sem[fp_id][gt_id] = gt_sem

                # only doors, windows, and frames for now
                if gt_sem not in [3, 4, 5]:
                    continue

                # this typically means there's a hole in this instance, skip for now
                if not len(gt_ring):
                    gt_skipped += 1
                    continue

                # this instance is not big enough, skip
                if np.sum(g_ins_full == gt_id) <= 10:
                    gt_skipped += 1
                    continue

                # get this instance's 32-length target vector
                gt_hist_dict = data_utils.get_four_way_hist(
                    g_ins_full, g_sem_full, gt_sem_rings, gt_id
                )

                if not gt_hist_dict:
                    gt_skipped += 1
                    continue

                self.gt_hist_dicts[fp_id][gt_id] = gt_hist_dict
                fp_examples.add((fp_id, gt_id, "gt"))

            # go through predicted instances first, appending both predicted and
            # corresponding GT instance examples
            for pred_id, (pred_sem, pred_ring) in pred_sem_rings.items():
                pred_id = int(pred_id)

                self.pred_ins_sem[fp_id][pred_id] = pred_sem

                # only doors, windows, and frames for now
                if pred_sem not in [3, 4, 5]:
                    continue

                # this typically means there's a hole in this instance, skip for now
                if not len(pred_ring):
                    pred_skipped += 1
                    continue

                # this instance is not big enough, skip
                if np.sum(p_ins_full == pred_id) <= 10:
                    pred_skipped += 1
                    continue

                pred_hist_dict = data_utils.get_four_way_hist(
                    p_ins_full, p_sem_full, pred_sem_rings, pred_id
                )

                if not pred_hist_dict:
                    pred_skipped += 1
                    continue

                self.pred_hist_dicts[fp_id][pred_id] = pred_hist_dict

                # make sure it overlaps at least 50% with a GT instance
                gt_id = self.pred2gt[fp_id][pred_id]

                if gt_id < 0:  # this means there's no valid GT correspondence
                    pred_skipped += 1
                    # data_utils.vis_bad_pair(pred_id, p_ins_full, p_sem_full, g_sem_full)
                    continue

                if gt_id not in self.gt_hist_dicts[fp_id].keys():
                    pred_skipped += 1
                    continue

                gt_sem = str(self.gt_ins_sem[fp_id][gt_id])
                assert int(gt_sem) == pred_sem

                pred_mask = p_ins_full == pred_id
                gt_mask = g_ins_full == gt_id
                iou = (
                    np.logical_and(pred_mask, gt_mask).sum()
                    / np.logical_or(pred_mask, gt_mask).sum()
                )

                if iou < 0.5:
                    pred_skipped += 1
                    continue

                fp_examples.add((fp_id, pred_id, "pred"))
                fp_examples.add((fp_id, gt_id, "gt"))

                if False:
                    data_utils.vis_pred_gt_pair(
                        pred_id,
                        gt_id,
                        p_ins_full,
                        p_sem_full,
                        g_ins_full,
                        g_sem_full,
                        pred_label,
                        gt_label,
                        "./gt_vis/%s_%d" % (fp_id, pred_id),
                    )

            self.examples.extend(list(fp_examples))

        print(
            "%d samples, %d GT skipped, %d pred skipped"
            % (len(self.examples), gt_skipped, pred_skipped)
        )

    def reshape_ok(self, ins_full, sem_full, ins_id):
        ins_crop, sem_crop, _ = data_utils.get_crop(ins_full, sem_full, ins_id)

        side_len = self.hparams["encoder_side_len"]
        ins_crop = utils.resize(ins_crop, (side_len, side_len))

        ins_mask = ins_crop == ins_id
        target_id = np.unique(ins_crop[ins_mask])

        if len(target_id) != 1:
            return False
        else:
            return True

    def __len__(self):
        return len(self.examples)

    def getitem_cnn(self, fp_id, target_id, gt_or_pred):
        if gt_or_pred == "gt":
            ins_full = self.gt_ins_full[fp_id]
            sem_full = self.gt_sem_full[fp_id]
        elif gt_or_pred == "pred":
            ins_full = self.pred_ins_full[fp_id]
            sem_full = self.pred_sem_full[fp_id]
        else:
            raise Exception

        # get the instance and semantic segmentation crops
        ins_crop, sem_crop, bbox = data_utils.get_crop(ins_full, sem_full, target_id)

        mini, minj, maxi, maxj = bbox
        img_crop = self.fp_img[fp_id][mini:maxi, minj:maxj]

        side_len = self.hparams["encoder_side_len"]
        ins_crop = utils.resize(ins_crop, (side_len, side_len))
        sem_crop = utils.resize(sem_crop, (side_len, side_len))
        img_crop = utils.resize(img_crop, (side_len, side_len), interp_mode.BILINEAR)

        # grab the label and instance masks of interest
        if gt_or_pred == "gt":
            hist_dict = self.gt_hist_dicts[fp_id][target_id]
        elif gt_or_pred == "pred":
            gt_id = self.pred2gt[fp_id][target_id]
            hist_dict = self.gt_hist_dicts[fp_id][gt_id]
        else:
            raise Exception

        label = data_utils.get_label(hist_dict)

        # random rotation augmentation
        if self.hparams["augmentation"]:
            k = np.random.choice([0, 1, 2, 3])
            img_crop = np.rot90(img_crop.copy(), k=k).copy()
            ins_crop = np.rot90(ins_crop.copy(), k=k).copy()
            sem_crop = np.rot90(sem_crop.copy(), k=k).copy()
            label = np.concatenate([label[2 * k :], label[: 2 * k]])

        # prep input to network
        img_crop = img_crop[np.newaxis, ...]
        ins_mask = (ins_crop == target_id).astype(np.float32)[np.newaxis, ...]
        sem_crop = data_utils.to_onehot(sem_crop, num_classes=8)
        combined = np.concatenate([sem_crop, img_crop, ins_mask], axis=0)

        # convert things to PyTorch tensors
        combined = torch.FloatTensor(combined)
        label = torch.FloatTensor(label)

        data_dict = {
            "fp_id": fp_id,
            "target_id": target_id,
            "ins_crop": ins_crop,
            "sem_crop": sem_crop,
            "combined": combined,
            "label": label,
        }

        return data_dict

    def getitem_tf(self, fp_id, target_id, gt_or_pred):
        raise Exception("Not doing this")

        if gt_or_pred == "gt":
            ins_full = self.gt_ins_full[fp_id]
            sem_full = self.gt_sem_full[fp_id]
        elif gt_or_pred == "pred":
            ins_full = self.pred_ins_full[fp_id]
            sem_full = self.pred_sem_full[fp_id]
        else:
            raise Exception

        # get the instance and semantic segmentation crops
        ins_crop, sem_crop, bbox = data_utils.get_crop(ins_full, sem_full, target_id)

        mini, minj, maxi, maxj = bbox
        img_crop = self.fp_img[fp_id][mini:maxi, minj:maxj][..., np.newaxis]

        side_len = self.hparams["encoder_side_len"]
        ins_crop = utils.resize(ins_crop, (side_len, side_len))
        sem_crop = utils.resize(sem_crop, (side_len, side_len))
        img_crop = utils.resize(img_crop, (side_len, side_len), interp_mode.BILINEAR)

        ins_ids = sorted(np.unique(ins_crop))

        # gather the instance masks and mark which one is the target
        ins_masks = []
        labels = []
        ignore_mask = []

        if gt_or_pred == "gt":
            fp_labels = self.gt_labels[fp_id]
            fp_ins_sem = self.gt_ins_sem[fp_id]
        elif gt_or_pred == "pred":
            fp_labels = self.pred_labels[fp_id]
            fp_ins_sem = self.pred_ins_sem[fp_id]
        else:
            raise Exception

        for ins_id in ins_ids:
            # add this instance's one-hot semantic crop
            ins_mask = ins_crop == ins_id

            h, w = ins_crop.shape
            ins_sem = fp_ins_sem[ins_id]
            one_hot_mask = np.zeros([h, w, self.hparams["n_classes"]], dtype=np.float32)
            one_hot_mask[:, :, ins_sem][ins_mask] = 1.0

            # concat the image to it
            # one_hot_mask = np.concatenate([one_hot_mask, img_crop], axis=2)

            ins_masks.append(one_hot_mask)

            # only compute loss on the target instance for now
            if ins_id == target_id:
                ignore_mask.append(False)

                # always train on GT topology
                if gt_or_pred == "gt":
                    labels.append(fp_labels[target_id])
                elif gt_or_pred == "pred":
                    gt_id = self.pred2gt[fp_id][target_id]
                    labels.append(self.gt_labels[fp_id][gt_id])
                else:
                    raise Exception

            else:
                ignore_mask.append(True)
                labels.append(fp_labels[target_id])

        ins_masks = np.stack(ins_masks)
        assert (sum(ignore_mask) + 1) == len(ignore_mask)

        # random rotation augmentation
        if self.hparams["augmentation"]:
            k = np.random.choice([0, 1, 2, 3])
            ins_crop = np.rot90(ins_crop, k=k).copy()
            sem_crop = np.rot90(sem_crop, k=k).copy()
            ins_masks = np.rot90(ins_masks, k=k, axes=(1, 2)).copy()
            labels = [np.concatenate((x[8 * k :], x[: 8 * k])) for x in labels]

        # convert things to PyTorch tensors
        ins_masks = torch.FloatTensor(ins_masks)
        labels = torch.FloatTensor(np.stack(labels))
        ignore_mask = torch.BoolTensor(ignore_mask)

        ins_masks = ins_masks.permute([0, 3, 1, 2])

        data_dict = {
            "ins_crop": ins_crop,
            "sem_crop": sem_crop,
            "ins_masks": ins_masks,
            "labels": labels,
            "ignore_mask": ignore_mask,
        }

        return data_dict

    def __getitem__(self, idx):
        if self.cnn_only:
            return self.getitem_cnn(*self.examples[idx])
        else:
            return self.getitem_tf(*self.examples[idx])


def check_class_imbalance():
    hparams = {
        "skip_uncommon": False,
        "uncommon_threshold": 20,
        "encoder_side_len": 224,
        "augmentation": False,
        "n_classes": 8,
    }
    class_map = data_utils.class_map

    my_dataset = FloorplanDataset(range(10), hparams, cnn_only=True)

    num_zeros = 0
    num_ones = 0

    for i, ex in enumerate(tqdm(my_dataset)):
        label = ex["label"]
        num_zeros += np.sum(label.numpy() == 0.0)
        num_ones += np.sum(label.numpy() == 1.0)

    print("Num zeros: %d" % num_zeros)
    print("Num ones: %d" % num_ones)


def main():
    hparams = {
        "skip_uncommon": False,
        "uncommon_threshold": 20,
        "encoder_side_len": 224,
        "augmentation": True,
        "n_classes": 8,
    }
    class_map = data_utils.class_map

    my_dataset = FloorplanDataset(range(10), hparams, cnn_only=True)

    for i, ex in enumerate(tqdm(my_dataset)):
        continue

        if i >= 100:
            break

        # NOTE this portion is for when we're training just a CNN
        sem_crop = ex["combined"][:8, :, :].argmax(0)
        fp_img = ex["combined"][8, :, :]
        ins_mask = ex["combined"][9, :, :]

        # if ins_mask.sum() >= 50:
        #   continue

        fig, [ax1, ax2] = plt.subplots(ncols=2)

        ax1.imshow(fp_img, cmap="gray")
        ax2.imshow(
            sem_crop / 7.0,
            cmap="nipy_spectral",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax2.imshow(ins_mask, cmap="gray", alpha=0.7)

        plt.title(str(ex["label"].numpy().tolist()))

        ax1.set_axis_off()
        ax2.set_axis_off()

        plt.tight_layout()
        plt.savefig(
            "./gt_vis/%s_%d.png" % (ex["fp_id"], ex["target_id"]),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()

        continue

        for j, (ins_mask, label, ignore) in enumerate(
            zip(ex["ins_masks"], ex["labels"], ex["ignore_mask"])
        ):
            ins_sem = max(np.unique(ins_mask.argmax(axis=0)))

            plt.imshow(
                ex["sem_crop"] / 7.0,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            plt.imshow(ins_mask.max(axis=0)[0], cmap="gray", alpha=0.7)

            top_str = ", ".join([class_map[a] for (a, b) in enumerate(top_hist) if b])
            right_str = ", ".join(
                [class_map[a] for (a, b) in enumerate(right_hist) if b]
            )
            bottom_str = ", ".join(
                [class_map[a] for (a, b) in enumerate(bottom_hist) if b]
            )
            left_str = ", ".join([class_map[a] for (a, b) in enumerate(left_hist) if b])

            target_or_not = "target\n" if (not ignore) else "meh\n"
            plt.title(
                target_or_not
                + "%s\n" % class_map[ins_sem]
                + "top: %s\n" % top_str
                + "right: %s\n" % right_str
                + "bottom: %s\n" % bottom_str
                + "left: %s\n" % left_str,
                fontsize=10,
            )

            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                "./gt_vis/%03d_%02d.png" % (i, j),
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


if __name__ == "__main__":
    main()
    # check_class_imbalance()
