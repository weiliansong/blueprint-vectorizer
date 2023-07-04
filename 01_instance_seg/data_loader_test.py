import os
import torch
import json
import numpy as np
import skimage
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.transform import resize
import matplotlib
import matplotlib.patches as mpatches
import torchvision.transforms.functional as TF

from utils.config import Struct, load_config, compose_config_str

# define the class mapping
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


class FloorplanDatasetTest(Dataset):
    def __init__(self, phase, test_fs, configs):
        super(FloorplanDatasetTest, self).__init__()

        if phase not in ["train", "test", "val"]:
            raise ValueError("invalid phase {}".format(phase))

        self.phase = phase
        self.configs = configs
        self.split_dir = os.path.join(
            configs.base_dir, configs.split_dir, "{}.json".format(phase)
        )
        self.augmentation = configs.augmentation
        self.h = configs.h
        self.w = configs.w
        self.method = configs.method
        self.pad_size = configs.pad_size
        self.margin_size = configs.margin_size
        self.n = configs.non_overlap
        self.test_fs = test_fs

        print("Evaluating on %d blueprints" % len(self.test_fs))

        if phase == "test" or phase == "val":
            self.augmentation = ""

    def __len__(self):
        return len(self.test_fs)

    def __getitem__(self, idx):
        test_f = self.test_fs[idx]

        image = Image.open(test_f)
        image = image.convert("L").convert("RGB")
        image = self.normalize(image)

        (
            image,
            image_full,
            pad_y1,
            pad_y2,
            pad_x1,
            pad_x2,
        ) = self.full_crops_dense(
            image,
            self.h,
            self.w,
            self.pad_size,
            self.margin_size,
            self.n,
        )

        image = torch.Tensor(image)
        image_full = torch.Tensor(image_full)

        image = image.permute([0, 1, 4, 2, 3])
        image_full = image_full.permute([2, 0, 1])

        pad = [pad_y1, pad_y2, pad_x1, pad_x2]
        floorplan_id = test_f.split(os.sep)[-1].split(".")[0]

        data = {
            "image": image,
            "image_full": image_full,
            "pad": pad,
            "f_id": floorplan_id,
        }

        return data

    def normalize(self, image):
        image = np.array(image, dtype=np.float32) / 255.0
        return image

    def transform(self, image, label):
        image = Image.fromarray((image * 255).astype(np.uint8))
        label = Image.fromarray((label).astype(np.uint8))

        # # rotate 0 or 90 or 180 or 270 degree
        # angle = np.random.randint(3)  # clock-wise rotation
        # angle = angle * 90
        # image = TF.rotate(image, angle)
        # label = TF.rotate(label, angle, fill=(0,))

        # apply flip
        # if np.random.random() > 0.5:

        # image = TF.hflip(image)
        # label = TF.hflip(label)

        image = TF.vflip(image)
        label = TF.vflip(label)

        image = self.normalize(np.asarray(image))
        label = np.asarray(label)

        return image, label

    def full_crop(self, image, label, h, w, pad_size):
        """
        utility function to resize sample(PIL image and label) to a given dimension
        without cropping information. the network takes in tensors with dimensions
        that are multiples of 32.

        :param img: numpy image to resize
        :param label: numpy array with the label to resize
        :param h: desired height
        :param w: desired width

        :return: the resized image, label
        """

        img_h, img_w, _ = image.shape

        # crop floorplan image to get rid of extra space around the floorplan
        miny, minx, maxy, maxx = self.get_region_bbox(label)
        # add small padding
        miny = max(miny - pad_size, 0)
        minx = max(minx - pad_size, 0)
        maxy = min(maxy + pad_size, img_h)
        maxx = min(maxx + pad_size, img_w)

        image = image[miny:maxy, minx:maxx, :]
        label = label[miny:maxy, minx:maxx]

        img_h, img_w, _ = image.shape

        pad_h = (max(img_h, img_w) - img_h) // 2
        pad_w = (max(img_h, img_w) - img_w) // 2

        # add padding to become a square
        image = np.pad(
            image,
            [
                ((max(img_h, img_w) - img_h) // 2, (max(img_h, img_w) - img_h) // 2),
                ((max(img_h, img_w) - img_w) // 2, (max(img_h, img_w) - img_w) // 2),
                (0, 0),
            ],
            mode="constant",
            constant_values=0.0,
        )

        label = np.pad(
            label,
            [
                ((max(img_h, img_w) - img_h) // 2, (max(img_h, img_w) - img_h) // 2),
                ((max(img_h, img_w) - img_w) // 2, (max(img_h, img_w) - img_w) // 2),
            ],
            mode="constant",
            constant_values=0.0,
        )

        img_h, img_w, _ = image.shape

        # resizing
        scale_size = (h, w)
        image = Image.fromarray((image * 255).astype(np.uint8))
        label = Image.fromarray((label).astype(np.uint8))

        image = image.resize(scale_size, Image.ANTIALIAS)
        label = label.resize(scale_size, Image.ANTIALIAS)

        image = self.normalize(np.asarray(image))
        label = np.asarray(label)

        pad_h = int(pad_h * (h / img_h))
        pad_w = int(pad_w * (w / img_w))

        return image, label, pad_h, pad_w

    # ----------------------------------------------------------------------------------------

    def full_crops1(self, image, label, semantic, semantic_pred, h, w, pad_size):
        # return all crops are needed for merging

        img_h, img_w, _ = image.shape

        # print(image.shape)
        # print(semantic_pred.shape)
        # # print(STOP)

        # crop floorplan image to get rid of extra space around the floorplan
        miny, minx, maxy, maxx = self.get_region_bbox(label)
        # add small padding
        miny = max(miny - pad_size, 0)
        minx = max(minx - pad_size, 0)
        maxy = min(maxy + pad_size, img_h)
        maxx = min(maxx + pad_size, img_w)

        image = image[miny:maxy, minx:maxx, :]
        label = label[miny:maxy, minx:maxx]
        semantic_pred = semantic_pred[miny:maxy, minx:maxx]
        semantic = semantic[miny:maxy, minx:maxx]

        img_h, img_w, _ = image.shape

        pad_size_y = 0  # 128 - (img_h % 128)
        pad_size_x = 0  # 128 - (img_w % 128)

        # add padding to become a multiplier of 128
        image = np.pad(
            image,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
                (0, 0),
            ],
            mode="constant",
            constant_values=0.0,
        )

        label = np.pad(
            label,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="constant",
            constant_values=0.0,
        )

        semantic_pred = np.pad(
            semantic_pred,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="constant",
            constant_values=0.0,
        )

        semantic = np.pad(
            semantic,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="constant",
            constant_values=0.0,
        )

        img_h, img_w, _ = image.shape

        h_num = int((img_h - 256) / 128 + 1)
        w_num = int((img_w - 256) / 128 + 1)

        images = np.zeros((h_num, w_num, h, w, 3))
        labels = np.zeros((h_num, w_num, h, w))

        for h_i in range(h_num):
            for w_i in range(w_num):
                minx = w_i * 128
                miny = h_i * 128
                maxx = minx + 256
                maxy = miny + 256
                images[h_i, w_i, :, :] = image[miny:maxy, minx:maxx, :]
                labels[h_i, w_i, :, :] = label[miny:maxy, minx:maxx]

        pad_y1, pad_y2 = int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)
        pad_x1, pad_x2 = int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)

        return (
            images,
            labels,
            image,
            label,
            semantic,
            semantic_pred,
            pad_y1,
            pad_y2,
            pad_x1,
            pad_x2,
        )

    # ----------------------------------------------------------------------------------------

    def full_crops(self, image, label, semantic, semantic_pred, h, w, pad_size):
        # return all crops are needed for merging

        pad_size = pad_size  # + 16 # + 48

        img_h, img_w, _ = image.shape

        # print(image.shape)
        # print(semantic_pred.shape)
        # # print(STOP)

        # crop floorplan image to get rid of extra space around the floorplan
        miny, minx, maxy, maxx = self.get_region_bbox(label)
        # add small padding
        miny = max(miny - pad_size, 0)
        minx = max(minx - pad_size, 0)
        maxy = min(maxy + pad_size, img_h)
        maxx = min(maxx + pad_size, img_w)

        image = image[miny:maxy, minx:maxx, :]
        label = label[miny:maxy, minx:maxx]
        semantic_pred = semantic_pred[miny:maxy, minx:maxx]
        semantic = semantic[miny:maxy, minx:maxx]

        img_h, img_w, _ = image.shape

        pad_size_y = 128 - (img_h % 128)  # + 128 # 0
        pad_size_x = 128 - (img_w % 128)  # + 128 # 0

        # add padding to become a multiplier of 128
        # image = np.pad(image, [(int(pad_size_y/2), pad_size_y - int(pad_size_y/2)), \
        #                        (int(pad_size_x/2), pad_size_x - int(pad_size_x/2)), (0, 0)], \
        #                         mode='constant', constant_values=0.0)
        #           (int(pad_size_x/2), pad_size_x - int(pad_size_x/2))], \
        #                         mode='constant', constant_values=0.0)

        # semantic_pred = np.pad(semantic_pred, [(int(pad_size_y/2), pad_size_y - int(pad_size_y/2)), \
        #                        (int(pad_size_x/2), pad_size_x - int(pad_size_x/2))], \
        #                         mode='constant', constant_values=0.0)

        # semantic = np.pad(semantic, [(int(pad_size_y/2), pad_size_y - int(pad_size_y/2)), \
        #                 (int(pad_size_x/2), pad_size_x - int(pad_size_x/2))], \
        #                 mode='constant', constant_values=0.0)

        image = np.pad(
            image,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
                (0, 0),
            ],
            mode="edge",
        )

        label = np.pad(
            label,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="edge",
        )

        semantic_pred = np.pad(
            semantic_pred,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="edge",
        )

        semantic = np.pad(
            semantic,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="edge",
        )

        img_h, img_w, _ = image.shape

        h_num = int((img_h - 256) / 128 + 1)
        w_num = int((img_w - 256) / 128 + 1)

        images = np.zeros((h_num, w_num, h, w, 3))
        labels = np.zeros((h_num, w_num, h, w))

        for h_i in range(h_num):
            for w_i in range(w_num):
                minx = w_i * 128
                miny = h_i * 128
                maxx = minx + 256
                maxy = miny + 256
                images[h_i, w_i, :, :] = image[miny:maxy, minx:maxx, :]
                labels[h_i, w_i, :, :] = label[miny:maxy, minx:maxx]

        pad_y1, pad_y2 = int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)
        pad_x1, pad_x2 = int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)

        return (
            images,
            labels,
            image,
            label,
            semantic,
            semantic_pred,
            pad_y1,
            pad_y2,
            pad_x1,
            pad_x2,
        )

    # ----------------------------------------------------------------------------------------
    #
    def full_crops_dense(self, image, h, w, pad_size, margin_size, n):
        img_h, img_w, _ = image.shape

        img_h, img_w, _ = image.shape

        pad_size_y = n - (img_h % n) + pad_size  # 224 # 256  # 0
        pad_size_x = n - (img_w % n) + pad_size  # 224 # 256 # 0

        pad_y1, pad_y2 = int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)
        pad_x1, pad_x2 = int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)

        image = np.pad(
            image,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
                (0, 0),
            ],
            mode="edge",
        )

        img_h, img_w, _ = image.shape

        h_num = int((img_h - 256) / n + 1)
        w_num = int((img_w - 256) / n + 1)

        images = np.zeros((h_num, w_num, h, w, 3))

        for h_i in range(h_num):
            for w_i in range(w_num):
                minx = w_i * n
                miny = h_i * n
                maxx = minx + 256
                maxy = miny + 256
                images[h_i, w_i, :, :] = image[miny:maxy, minx:maxx, :]

        pad_y1, pad_y2 = int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)
        pad_x1, pad_x2 = int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)

        return (
            images,
            image,
            pad_y1,
            pad_y2,
            pad_x1,
            pad_x2,
        )

    # ----------------------------------------------------------------------------------------
    #
    def get_region_bbox(self, region_mask):
        yy, xx = np.nonzero(region_mask > 0)

        minx = int(np.min(xx))
        miny = int(np.min(yy))
        maxx = int(np.max(xx))
        maxy = int(np.max(yy))

        return [miny, minx, maxy, maxx]

    def semantic_pred_aligned(
        self, image, label_instance, label_semantic, crop_size=1024
    ):
        # uncrop and unpad semantic pred!!!

        # find the bounding box of the label annotation, plus a small padding
        # ii_semantic, jj_semantic = np.nonzero(label_semantic)
        ii_instance, jj_instance = np.nonzero(label_instance)

        ii = ii_instance
        jj = jj_instance

        border_size = 20  # leave a small border around the crop
        mini = max(ii.min() - border_size, 0)
        minj = max(jj.min() - border_size, 0)
        maxi = min(ii.max() + border_size, image.shape[0])
        maxj = min(jj.max() + border_size, image.shape[1])

        # crop the floorplans with the bounding box
        image_crop = image[mini:maxi, minj:maxj]
        # label_semantic_crop = label_semantic[mini:maxi, minj:maxj]
        label_instance_crop = label_instance[mini:maxi, minj:maxj]

        ## mini, maxi, minj, maxj

        # pad the crops to squares
        img_h, img_w = label_instance_crop.shape

        if img_h > img_w:
            pad_before = (img_h - img_w) // 2
            pad_after = img_h - pad_before - img_w
            pad_width = np.array([[0, 0], [pad_before, pad_after]])

        else:
            pad_before = (img_w - img_h) // 2
            pad_after = img_w - pad_before - img_h
            pad_width = np.array([[pad_before, pad_after], [0, 0]])

        # image_padded = np.pad(image_crop, pad_width, mode='constant')
        # label_semantic_padded = np.pad(label_semantic_crop,
        #                                 pad_width,
        #                                 mode='constant')
        label_instance_padded = np.pad(label_instance_crop, pad_width, mode="constant")

        w, h = label_instance_padded.shape
        ratio = label_instance_padded.shape[0]

        # reshape the two padded crops to the network input shape
        # image_padded = resize(image_padded, [crop_size, crop_size])

        label_instance_padded = resize(
            label_instance_padded,
            [crop_size, crop_size],
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )

        # ----------------------------------------------------
        # undo and aligning!

        label_semantic = np.argmax(label_semantic, axis=0)

        label_semantic_aligned = resize(
            label_semantic,
            [ratio, ratio],
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )

        i1 = pad_width[0, 0]
        i2 = -1 * pad_width[0, 1]

        j1 = pad_width[1, 0]
        j2 = -1 * pad_width[1, 1]

        if i2 == 0:
            i2 = None

        if j2 == 0:
            j2 = None

        label_semantic_aligned = label_semantic_aligned[i1:i2, j1:j2]

        i1 = mini
        j1 = minj
        i2 = label_instance.shape[0] - maxi
        j2 = label_instance.shape[1] - maxj

        pad_size = np.array([[i1, i2], [j1, j2]])
        label_semantic_aligned = np.pad(
            label_semantic_aligned, pad_size, mode="constant"
        )

        return label_semantic_aligned

    # ----------------------------------------------------------------------------------------
    def full_crops_check(self, image, label, semantic, semantic_pred, h, w, pad_size):
        img_h, img_w, _ = image.shape

        ratio = 1.5
        new_img_h = int(img_h / ratio)
        new_img_w = int(img_w / ratio)

        miny, minx, maxy, maxx = self.get_region_bbox(label)
        # ----------
        # miny, minx, maxy, maxx = int(miny/ratio), int(minx/ratio), int(maxy/ratio), int(maxx/ratio)

        # label = self.normalize(label)
        # image = resize(image, [new_img_h, new_img_w], anti_aliasing=True)
        # label = resize(label, [new_img_h, new_img_w], anti_aliasing=True)
        # semantic = resize(semantic, [new_img_h, new_img_w], anti_aliasing=True)
        # semantic_pred = resize(semantic_pred, [new_img_h, new_img_w], anti_aliasing=True)
        # label = label*255.0
        # label = label.astype(int)
        # print(np.unique(label))
        # print(lll)

        # ----------

        # label_semantic_padded = resize(label_semantic_padded,
        #                            [self.crop_size, self.crop_size],
        #                            order=0,
        #                            preserve_range=True,
        #                            anti_aliasing=False)

        # return all crops are needed for merging

        img_h, img_w, _ = image.shape

        # print(image.shape)
        # print(semantic_pred.shape)
        # # print(STOP)

        # change
        pad_size_yy = pad_size + 32  # 128+64 # - 8
        pad_size_xx = pad_size

        # crop floorplan image to get rid of extra space around the floorplan
        # miny, minx, maxy, maxx = self.get_region_bbox(label)
        # add small padding
        miny = max(miny - pad_size_yy, 0)
        minx = max(minx - pad_size_xx, 0)
        maxy = min(maxy + pad_size_yy, img_h)
        maxx = min(maxx + pad_size_xx, img_w)

        image = image[miny:maxy, minx:maxx, :]
        label = label[miny:maxy, minx:maxx]
        semantic_pred = semantic_pred[miny:maxy, minx:maxx]
        semantic = semantic[miny:maxy, minx:maxx]

        # image = np.fliplr(image)

        img_h, img_w, _ = image.shape

        # change
        pad_size_y = 0  # 128 - (img_h % 128) # + 32 # + 64 # + 128
        pad_size_x = 0  # 128 - (img_w % 128) # + 128

        # add padding to become a multiplier of 128
        image = np.pad(
            image,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
                (0, 0),
            ],
            mode="constant",
            constant_values=0.0,
        )

        label = np.pad(
            label,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="constant",
            constant_values=0.0,
        )

        semantic_pred = np.pad(
            semantic_pred,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="constant",
            constant_values=0.0,
        )

        semantic = np.pad(
            semantic,
            [
                (int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)),
                (int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)),
            ],
            mode="constant",
            constant_values=0.0,
        )

        img_h, img_w, _ = image.shape

        # change
        overlap = 128  # 128

        h_num = int((img_h - 256) / overlap + 1)
        w_num = int((img_w - 256) / overlap + 1)

        images = np.zeros((h_num, w_num, h, w, 3))
        labels = np.zeros((h_num, w_num, h, w))

        for h_i in range(h_num):
            for w_i in range(w_num):
                minx = w_i * overlap
                miny = h_i * overlap
                maxx = minx + 256
                maxy = miny + 256
                images[h_i, w_i, :, :] = image[miny:maxy, minx:maxx, :]
                labels[h_i, w_i, :, :] = label[miny:maxy, minx:maxx]

        pad_y1, pad_y2 = int(pad_size_y / 2), pad_size_y - int(pad_size_y / 2)
        pad_x1, pad_x2 = int(pad_size_x / 2), pad_size_x - int(pad_size_x / 2)

        return (
            images,
            labels,
            image,
            label,
            semantic,
            semantic_pred,
            pad_y1,
            pad_y2,
            pad_x1,
            pad_x2,
        )

    # ----------------------------------------------------------------------------------------


# check dataloder ...
if __name__ == "__main__":
    config_dict = load_config(file_path="utils/config.yaml")
    configs = Struct(**config_dict)

    train_dataset = FloorplanDatasetTest(phase="test", configs=configs)

    for idx, batch_data in enumerate(train_dataset):
        if idx == 1:
            image_full = batch_data["image_full"]
            label_full = batch_data["label_full"]
            semantic_pred_full = batch_data["semantic_pred_full"]
            semantic_full = batch_data["semantic_full"]

            image_full = image_full.permute([1, 2, 0])

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, dpi=150)

            # don't visualize background as a 0 label
            # label_full = np.ma.masked_where(label_full == 0, label_full)

            ax1.imshow(image_full)
            ax1.set_axis_off()

            ax2.imshow(image_full)
            ax2.imshow(label_full, cmap="nipy_spectral", alpha=0.7)
            ax2.set_axis_off()

            ax3.imshow(image_full)
            ax3.imshow(semantic_pred_full, cmap="nipy_spectral", alpha=0.7)
            ax3.set_axis_off()

            ax4.imshow(image_full)
            ax4.imshow(semantic_full, cmap="nipy_spectral", alpha=0.7)
            ax4.set_axis_off()

            plt.tight_layout()
            plt.show()
            plt.close()

            nipy_cmap = matplotlib.cm.get_cmap("nipy_spectral")

            bg = mpatches.Patch(color=nipy_cmap(0.0 / 6.0), label="Background")
            outer = mpatches.Patch(color=nipy_cmap(1.0 / 6.0), label="Outer Wall")
            inner = mpatches.Patch(color=nipy_cmap(2.0 / 6.0), label="Inner Wall")
            window = mpatches.Patch(color=nipy_cmap(3.0 / 6.0), label="Window")
            door = mpatches.Patch(color=nipy_cmap(4.0 / 6.0), label="Door")
            frame = mpatches.Patch(color=nipy_cmap(5.0 / 6.0), label="Frame")
            room = mpatches.Patch(color=nipy_cmap(6.0 / 6.0), label="Room")

            handles = [bg, outer, inner, window, door, frame, room]
            plt.legend(handles=handles, fontsize=2, loc=1)

            plt.axis("off")
            plt.tight_layout()
            plt.show()
            # plt.savefig(VIS_ROOT + '%s.png' % floorplan_id)
            plt.close()
            # print(wait)
