import argparse
import glob
import json
import os
import shutil
import time
from os.path import join as pj

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paths
import torch
import torch.nn as nn
import torchvision
import utils
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
from skimage.morphology import dilation, erosion, square
from skimage.transform import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))

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


""" Startup Stuff :P """
args = utils.parse_arguments()
config = utils.parse_config(args.config_path)
model_dir = "../ckpts/02_semantic_class/%d/" % config["test_id"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(model_dir + "model.pt"):
    raise Exception("Model not trained")

if args.restart and os.path.exists(paths.PRED_SEMANTIC_ROOT):
    shutil.rmtree(paths.PRED_SEMANTIC_ROOT)
os.makedirs(paths.PRED_SEMANTIC_ROOT, exist_ok=True)

vis_root = paths.GA_ROOT + "preprocess/visualize_pred/"
if args.restart and os.path.exists(vis_root):
    shutil.rmtree(vis_root)
os.makedirs(vis_root, exist_ok=True)


""" Model """
model = torchvision.models.resnet50(num_classes=7)
model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.to(device)
model.load_state_dict(torch.load(model_dir + "model_best.pt"))
model = model.eval()


""" Stuff happens here :D """
if args.test_folder:
    print("User-specified test folder")
    img_files = glob.glob(pj(args.test_folder, "*"))
    fp_ids = [x.split("/")[-1].split(".")[0] for x in img_files]
else:
    split_json = paths.SPLITS_ROOT + "ids_%d.json" % config["test_id"]
    with open(split_json, "r") as f:
        fp_ids = json.load(f)

for fp_id in tqdm(fp_ids):
    image = Image.open(paths.IMG_ROOT + "%s.jpg" % fp_id)
    image = np.array(image, dtype=np.float32) / 255.0

    instance_pred = np.load(paths.PRED_INSTANCE_ROOT + "%s.npy" % fp_id)
    semantic_pred = np.zeros_like(instance_pred)

    for instance_id in np.unique(instance_pred):
        instance_mask = instance_pred == instance_id

        if instance_id == 0:
            pass

        else:
            ii, jj = np.nonzero(instance_mask)
            seeds = np.array(list(zip(ii, jj)))

            seed_idx = np.random.choice(
                range(len(seeds)), size=min(20, len(seeds)), replace=False
            )
            seeds = seeds[seed_idx]

            averaged_pred = torch.zeros([1, 7], dtype=torch.float32)

            for center_i, center_j in seeds:
                side_len = 224
                mini = max(center_i - side_len // 2, 0)
                minj = max(center_j - side_len // 2, 0)
                mini = min(mini, instance_pred.shape[0] - side_len)
                minj = min(minj, instance_pred.shape[1] - side_len)
                maxi = min(mini + side_len, instance_pred.shape[0])
                maxj = min(minj + side_len, instance_pred.shape[1])

                assert (mini >= 0) and (minj >= 0)
                assert (maxi <= instance_pred.shape[0]) and (
                    maxj <= instance_pred.shape[1]
                )
                assert ((maxi - mini) == side_len) and ((maxj - minj) == side_len)

                image_crop = np.copy(image[mini:maxi, minj:maxj])
                instance_crop = np.copy(instance_pred[mini:maxi, minj:maxj])
                crop_mask = (instance_crop == instance_id).astype(np.float32)

                image_crop = torch.Tensor(image_crop)
                crop_mask = torch.Tensor(crop_mask)
                combined = torch.stack([image_crop, crop_mask], axis=0)

                combined = combined.unsqueeze(0)
                combined = combined.to(device)
                pred = model(combined).detach().cpu()
                averaged_pred += nn.functional.softmax(pred, dim=1) / len(seeds)

            # NOTE background is not a class, so +1 to compensate for that
            instance_pred_sem = averaged_pred.argmax().numpy() + 1
            semantic_pred[instance_mask] = instance_pred_sem

    # save predicted semantic segmentations
    np.save(paths.PRED_SEMANTIC_ROOT + "%s.npy" % fp_id, semantic_pred)

    # save full visualizations
    instance_img = random_cmap(instance_pred) * 255.0
    sem_pred_img = my_cmap(semantic_pred / 7.0) * 255.0

    full_vis = np.concatenate([instance_img, sem_pred_img], axis=1)
    full_vis = Image.fromarray(full_vis.astype("uint8"))

    # write some captions on it
    # height, width, _ = instance_img.shape
    # font = ImageFont.truetype("SourceCodePro-Regular.ttf", 32)

    # drawer = ImageDraw.Draw(full_vis)
    # drawer.text((5 + width * 0, 5), "Instance Pred", font=font, fill=(0, 0, 0))
    # drawer.text((5 + width * 1, 5), "Semantic Vote", font=font, fill=(0, 0, 0))
    # drawer.text((5 + width * 2, 5), "Instance Pred", font=font, fill=(0, 0, 0))
    # drawer.text((5 + width * 3, 5), "Semantic GT", font=font, fill=(0, 0, 0))

    full_vis.save(vis_root + "%s.png" % fp_id)
