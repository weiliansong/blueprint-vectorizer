import glob
import os
import random
from os.path import join as pj

import metric_loss
import numpy as np
import torch
import torch.nn as nn
from data_loader_test import FloorplanDatasetTest
from imageio import imread, imsave
from models.github_model import *
from models.unet import UNet
from models.unet.unet_model import UNet
from sklearn.decomposition import PCA
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils.config import Struct, compose_config_str, load_config
from utils.misc import count_parameters, save_checkpoint, transfer_optimizer_to_gpu
from vis_cat import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_arguments()
config_dict = load_config(file_path="utils/config.yaml")
configs = Struct(**config_dict)

if configs.seed:
    torch.manual_seed(configs.seed)
    if configs.use_cuda:
        torch.cuda.manual_seed(configs.seed)
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    print("set random seed to {}".format(configs.seed))


model = UNet(configs.channel, configs.embedding_dim)

model_path = configs.exp_base_dir + "/%d/checkpoint_39.pth.tar" % args.test_fold_id
if model_path:
    # checkpoint = torch.load(configs.model_path)
    # checkpoint = torch.load(configs.model_path, map_location = device).cuda()
    checkpoint = torch.load(
        model_path, map_location=device
    )  # , map_location=lambda storage, loc: storage)
    # checkpoint = torch.load(configs.model_path, map_location = device)
    # checkpoint = torch.load(configs.model_path, map_location = {'cuda:0': 'cpu'})

    model.load_state_dict(checkpoint["state_dict"])
    epoch_num = checkpoint["epoch"]
    print("=> loaded checkpoint {} (epoch {})".format(model_path, epoch_num))

model.to(device)
model.eval()

# retrieve all the test images
# TODO implement
if args.test_folder:
    test_fs = glob.glob(pj(args.test_folder, "*"))

test_dataset = FloorplanDatasetTest(
    "test",
    test_fs=test_fs,
    configs=configs,
)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

flag_reset = True  # True
n = configs.non_overlap
f_ids = []

if flag_reset:
    with torch.no_grad():
        for idx, batch_data in enumerate(test_dataset):
            # torch.cuda.empty_cache()
            f_id = batch_data["f_id"]
            check_path = configs.base_dir + "/preprocess/boundary_pred/{}.npy".format(
                f_id
            )
            if os.path.exists(check_path):
                print("Skipping %s" % f_id)
                continue
            if not os.path.exists(os.path.dirname(check_path)):
                os.makedirs(os.path.dirname(check_path), exist_ok=True)

            images = batch_data["image"].to(device)
            image_full = batch_data["image_full"]
            pad = batch_data["pad"]

            f_ids.append(f_id)

            h_num = images.size(0)
            w_num = images.size(1)

            # print(image_full.shape)
            _, h, w = image_full.shape

            h_num1 = h_num
            pred_segs = np.zeros((h_num1, w_num, 256, 256))
            pred_segs1 = np.zeros((h_num1, w_num, 256, 256))

            counter = 0
            # check_path = "Data/final_index{}.npy".format(f_id)
            # print(check_path)
            # if not os.path.exists(check_path): # idx == 8:
            if True:  # idx == 8:
                for h_i in range(h_num):
                    for w_i in range(w_num):
                        py = h_i * n
                        px = w_i * n

                        pye = py + 255
                        pxe = px + 255

                        pad_new = [
                            int(py < pad[0]) * (pad[0] - py),
                            int(pye > h - pad[1]) * (pye - h + pad[1]),
                            int(px < pad[2]) * (pad[2] - px),
                            int(pxe > w - pad[3]) * (pxe - w + pad[3]),
                        ]

                        image = images[h_i, w_i, :, :, :]
                        image = torch.unsqueeze(image, 0)

                        pred = model(image)
                        image = image.cpu()
                        pred = pred.cpu()

                        counter = counter + 1
                        print(
                            "predict sample {}: {}/{}".format(
                                idx, counter, h_num * w_num
                            )
                        )

                        pred = pred.detach().cpu()[0]

                        # --------------------------------------------

                        image = np.pad(
                            image,
                            [(0, 0), (0, 0), (0, 1), (0, 1)],
                            mode="constant",
                            constant_values=0,
                        )
                        pred = np.pad(
                            pred,
                            [(0, 0), (0, 1), (0, 1)],
                            mode="constant",
                            constant_values=0,
                        )

                        image = image[
                            :,
                            :,
                            pad_new[0] : -pad_new[1] - 1,
                            pad_new[2] : -pad_new[3] - 1,
                        ]
                        pred = pred[
                            :,
                            pad_new[0] : -pad_new[1] - 1,
                            pad_new[2] : -pad_new[3] - 1,
                        ]

                        image = torch.from_numpy(image)
                        pred = torch.from_numpy(pred)

                        pred_seg = segment(image, pred, "new", idx)
                        pred_seg = np.pad(
                            pred_seg,
                            [(pad_new[0], pad_new[1]), (pad_new[2], pad_new[3])],
                            mode="constant",
                            constant_values=0.0,
                        )

                        pred_seg1 = np.copy(pred_seg)
                        pred_segs1[h_i, w_i, :, :] = pred_seg1

                        pred_seg = process(pred_seg)
                        # viz_image(pred_seg, h_i*w_num+w_i)

                        pred_segs[h_i, w_i, :, :] = pred_seg

                # remove padding
                image_full = image_full[:, pad[0] : -pad[1], pad[2] : -pad[3]]

                images = images.cpu()
                _, h, w = image_full.shape

                boundary = final_seg_merge(pred_segs1, f_id, pad, h, w, n, configs)

                print("=======================================")


print("Data saved!")
print("DONE!")
