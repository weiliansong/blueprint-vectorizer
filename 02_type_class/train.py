import matplotlib

matplotlib.use("Agg")

import os
import shutil
import argparse

import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

import utils
import paths
import data_loader


""" Startup Stuff :P """
args = utils.parse_arguments()
config = utils.parse_config(args.config_path)
model_dir = "../ckpts/02_semantic_class/%d/" % config["test_id"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.restart:
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
os.makedirs(model_dir + "cm/", exist_ok=True)


""" Dataloader """
train_ids = [split_id for split_id in range(10) if split_id != config["test_id"]]
val_ids = [
    train_ids.pop(),
]

print("test", config["test_id"])
print("train", train_ids)
print("val", val_ids)

train_dataset = data_loader.FloorplanSegmentation(train_ids)
val_dataset = data_loader.FloorplanSegmentation(val_ids)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=False,
)


""" Model """
model = torchvision.models.resnet50(num_classes=7)
model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.to(device)


""" Loss """
# class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32)
# class_weights = class_weights.to(device)
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)


""" Optimizer and Scheduler """
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=config["step_size"], gamma=config["gamma"]
)


""" Used for validation later on """
best_epoch_loss = float("inf")
current_patience = 0


""" Tensorboard summary writer """
writer = SummaryWriter(log_dir=model_dir)


""" Stuff happens here :D """
for i_epoch in range(config["num_epochs"]):
    print("\n>>> Epoch %d" % (i_epoch + 1))

    # Train 1 epoch
    model = model.train()

    train_data_iter = tqdm(
        train_dataloader, total=len(train_dataloader), desc="train: "
    )

    for i_batch, batch_data in enumerate(train_data_iter):
        images, labels = batch_data

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(images)
        train_loss = criterion(preds, labels)

        train_loss.backward()
        optimizer.step()

        train_data_iter.set_postfix(loss=float(train_loss))

        num_iter = i_epoch * len(train_dataloader) + i_batch
        writer.add_scalar("Loss/train", train_loss, num_iter)

    # step the LR
    scheduler.step()
    if (i_epoch + 1) % config["step_size"] == 0:
        current_patience = 0
        print("\n[!] LR stepped, current LR: %f" % scheduler.get_last_lr()[0])

    torch.save(model.state_dict(), model_dir + "model.pt")

    # start running validation
    model = model.eval()
    val_losses = []
    val_preds = []
    val_labels = []

    val_data_iter = tqdm(val_dataloader, total=len(val_dataloader), desc="val: ")

    with torch.no_grad():
        for i_batch, batch_data in enumerate(val_data_iter):
            images, labels = batch_data

            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            val_loss = criterion(preds, labels)

            val_losses.append(val_loss)

            val_data_iter.set_postfix(loss=float(val_loss))

            val_preds.extend(preds.argmax(1).detach().cpu().numpy().tolist())
            val_labels.extend(labels.cpu().numpy().tolist())

    # save out confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    class_names = ["outer", "inner", "window", "door", "frame", "room", "symbol"]
    utils.plot_confusion_matrix(cm, class_names)
    plt.savefig(model_dir + "cm/%03d.png" % i_epoch)
    plt.close()

    # compare with best epoch loss, and see if we reached our patience
    epoch_loss = sum(val_losses) / len(val_losses)
    writer.add_scalar("Loss/val", epoch_loss, i_epoch)

    if epoch_loss < best_epoch_loss:
        best_epoch_loss = epoch_loss
        current_patience = 0

        # save model since this one is better
        torch.save(model.state_dict(), model_dir + "model_best.pt")

    else:
        current_patience += 1

    print(
        "\nval loss: %f | best loss: %f | patience: %d / %d"
        % (epoch_loss, best_epoch_loss, current_patience, config["patience"])
    )

    if current_patience == config["patience"]:
        print("\n\nPatience reached, early-stopping")
        break


print("\n\nfinished")
