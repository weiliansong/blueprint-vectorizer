import matplotlib

matplotlib.use("Agg")

import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import utils
from combined_loader import FloorplanDataset
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

"""
  Startup
"""
args = utils.parse_arguments()
hparams = utils.parse_yaml(args.hparam_f)
assert not (args.restart == True and args.resume == True)

# get the git commit hash
commit_hash = subprocess.getoutput("git rev-parse HEAD")
hparams["commit_hash"] = commit_hash

save_root = "../ckpts/03_frame_detect/%s/" % hparams["experiment_name"]

if args.restart and os.path.exists(save_root):
    shutil.rmtree(save_root)

os.makedirs(save_root, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# save hyparparameter config
yaml = YAML()
out = Path(save_root + "hparams.yaml")
yaml.dump(hparams, out)


"""
  Dataloader setup
"""
train_ids = [split_id for split_id in range(10) if split_id != hparams["test_id"]]
val_ids = [
    train_ids.pop(),
]

print("test", hparams["test_id"])
print("train", train_ids)
print("val", val_ids)

fp_dataset_train = FloorplanDataset(train_ids, hparams, cnn_only=True)
fp_loader_train = torch.utils.data.DataLoader(
    fp_dataset_train,
    batch_size=hparams["b_size"],
    shuffle=True,
    num_workers=hparams["n_cpu"],
)

fp_dataset_val = FloorplanDataset(val_ids, hparams, cnn_only=True)
fp_loader_val = torch.utils.data.DataLoader(
    fp_dataset_val,
    batch_size=hparams["b_size"],
    shuffle=True,
    num_workers=hparams["n_cpu"],
)


"""
  Loss, model, optimizer, tensorboard setup
"""
criterion = nn.BCEWithLogitsLoss()
criterion.to(device)

model = torchvision.models.resnet50(num_classes=8)
model.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr_backbone"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams["lr_drop"])

writer = SummaryWriter(save_root)


"""
  Resume if needed
"""
if args.resume:
    ckpt_f = save_root + "last.pth"
    assert os.path.exists(ckpt_f)

    print("Resuming from %s" % ckpt_f)
    last_ckpt = torch.load(ckpt_f)

    model.load_state_dict(last_ckpt["model"])
    optimizer.load_state_dict(last_ckpt["optimizer"])
    epoch_start = last_ckpt["epoch"] + 1

else:
    epoch_start = 0


"""
  Training
"""
batch_loss = 0
optimizer.zero_grad()
start = time.time()
best_val_loss = float("inf")

for epoch_i in range(epoch_start, hparams["n_epochs"]):
    print("\n>>> Epoch %d" % (epoch_i + 1))

    # training
    model.train()

    train_data_iter = tqdm(fp_loader_train, total=len(fp_loader_train), desc="train: ")

    for batch_i, batch in enumerate(train_data_iter):
        n_iter = epoch_i * len(fp_loader_train) + batch_i

        # also remove batch dimension
        combined = batch["combined"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        preds = model(combined)

        train_loss = criterion(input=preds, target=labels)
        train_loss.backward()
        optimizer.step()

        train_data_iter.set_postfix(loss=float(train_loss))

        writer.add_scalar("loss", train_loss, n_iter)

    scheduler.step()

    # validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_i, batch in enumerate(fp_loader_val):
            # also remove batch dimension
            combined = batch["combined"].to(device)
            labels = batch["label"].to(device)

            preds = model(combined)

            loss = criterion(input=preds, target=labels)
            val_loss += loss.data

    val_loss /= len(fp_loader_val)

    print("current: %.3f   best: %.3f" % (val_loss, best_val_loss))
    with open(save_root + "val_loss.csv", "a") as f:
        f.write("%d,%f\n" % (epoch_i, val_loss))

    # save model with best validation loss
    if val_loss < best_val_loss:
        print("better, saving...")
        best_val_loss = val_loss

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
            },
            save_root + "best.pth",
        )

    # save model after each epoch
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_i,
        },
        save_root + "last.pth",
    )
