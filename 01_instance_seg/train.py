import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from imageio import imsave, imread
import matplotlib.pyplot as plt
import random

from models.unet import UNet
from models.github_model import *
import metric_loss
from data_loader import FloorplanDataset
from models.unet.unet_model import UNet
from utils.misc import save_checkpoint, count_parameters, transfer_optimizer_to_gpu
from utils.config import Struct, load_config, compose_config_str, parse_arguments


def evaluate_model(model, val_loader, criterion):
    """
    evaluates average loss of a model on a given loss function, and average dice distance of
    some segmentations.
    :param model: the model to use for evaluation
    :param dataloader: a dataloader with the validation set
    :param loss_fn:
    :return: average loss, average dice distance
    """
    # start running validation

    running_loss = 0

    with torch.no_grad():
        for iter_i, batch_data in enumerate(val_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"]  # .to(device) # remove .to(device)
            preds = model(images)
            # loss = criterion(preds, labels)
            loss = criterion(preds.cpu(), labels)
            running_loss += loss.item()

    # compare with best epoch loss, and see if we reached our patience
    epoch_loss = running_loss / (iter_i + 1)

    return epoch_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_arguments()
config_dict = load_config(file_path="utils/config.yaml")
configs = Struct(**config_dict)
config_str = compose_config_str(configs, keywords=["lr", "batch_size"])
exp_dir = os.path.join(configs.exp_base_dir, "%d/" % args.test_fold_id)
configs.exp_dir = exp_dir

ckpt_save_path = os.path.join(configs.exp_dir)
if not os.path.exists(ckpt_save_path):
    os.mkdir(ckpt_save_path)

if configs.seed:
    torch.manual_seed(configs.seed)
    if configs.use_cuda:
        torch.cuda.manual_seed_all(configs.seed)
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    print("set random seed to {}".format(configs.seed))


# Dataloader
train_dataset = FloorplanDataset("train", args.test_fold_id, configs=configs)
# val_dataset = FloorplanDataset(phase='val', configs = configs)

train_loader = DataLoader(
    train_dataset,
    batch_size=configs.batch_size,
    num_workers=configs.num_workers,
    shuffle=True,
)
# val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, num_workers=configs.num_workers, shuffle=True)


# model
# model = FeatureExtractor(embedding_dim, context=context)
model = UNet(configs.channel, configs.embedding_dim)

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)


num_parameters = count_parameters(model)
print("total number of trainable parameters is: {}".format(num_parameters))


criterion = metric_loss.metricLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=configs.lr, weight_decay=configs.decay_rate
)
scheduler = StepLR(optimizer, step_size=configs.lr_step, gamma=configs.lr_gamma)

start_epoch = 0

if configs.resume:
    if os.path.isfile(configs.model_path):
        print("=> loading checkpoint '{}'".format(configs.model_path))
        checkpoint = torch.load(configs.model_path)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        if configs.use_cuda:
            transfer_optimizer_to_gpu(optimizer)
        print(
            "=> loaded checkpoint {} (epoch {})".format(configs.model_path, start_epoch)
        )
    else:
        print("no checkpoint found at {}".format(configs.model_path))

ckpt_save_path = os.path.join(configs.exp_dir)
if not os.path.exists(ckpt_save_path):
    os.mkdir(ckpt_save_path)

model.to(device)
model.train()

best_loss = np.Inf
best_train_loss = np.Inf
current_patience = 0

print("start")

for epoch_num in range(start_epoch, configs.max_epoch_num):
    print("learning_rate: :{:.6f}".format(optimizer.param_groups[0]["lr"]))
    start = time.time()

    running_loss = 0
    for iter_i, batch_data in enumerate(train_loader):

        images = batch_data["image"].to(device)
        labels = batch_data["label"]  # .to(device) # remove .to(device)

        optimizer.zero_grad()

        preds = model(images)

        # loss = criterion(preds, labels)
        loss = criterion(preds.cpu(), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if iter_i % 10 == 0:
            end = time.time()
            print(
                "Epoch: [{}][{}/{}]\t Loss: {:.4f} Time: {:.2f}".format(
                    epoch_num, iter_i, len(train_loader), loss.item(), end - start
                )
            )
            start = time.time()

        # add this part later
        # if iter_i % configs.visualize_iter == 0:
        #     print("HHH")

    scheduler.step()
    num_batches = iter_i + 1
    train_loss = running_loss / num_batches
    print(
        "****** epoch: [{}/{}], train loss:{:.4f}".format(
            epoch_num + 1, configs.max_epoch_num, train_loss
        )
    )

    model.train()

    if epoch_num % configs.val_interval == 0:
        save_checkpoint(
            {
                "epoch": epoch_num + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            checkpoint=ckpt_save_path,
            filename="checkpoint_{:02d}.pth.tar".format(epoch_num),
        )

    if train_loss <= best_train_loss:
        best_train_loss = train_loss
        print("best train loss:", best_train_loss)
        save_checkpoint(
            {
                "epoch": epoch_num + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            checkpoint=ckpt_save_path,
            filename="best_train_loss.pth.tar",
        )
