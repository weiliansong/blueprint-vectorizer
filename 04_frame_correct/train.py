import os
import time
import glob
import shutil
import subprocess

from pathlib import Path

import torch
import numpy as np

from ruamel.yaml import YAML
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import utils

from data_loader import FloorplanGraphDataset, floorplan_collate_fn


"""
  Startup
"""
args = utils.parse_arguments()
hparams = utils.parse_config(args.hparam_f)

# get the git commit hash
commit_hash = subprocess.getoutput("git rev-parse HEAD")
hparams["commit_hash"] = commit_hash

exp_name = hparams["experiment_name"]

root_folder = "../ckpts/04_frame_correct/%s/" % exp_name

assert not (args.restart == True and args.resume == True)

if args.restart and os.path.exists(root_folder):
    shutil.rmtree(root_folder)
os.makedirs(root_folder, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# save hyparparameter config
yaml = YAML()
out = Path(root_folder + "hparams.yaml")
yaml.dump(hparams, out)


"""
  Dataloader setup
"""
train_ids = [split_id for split_id in range(10) if split_id != hparams["test_id"]]

print("test", hparams["test_id"])
print("train", train_ids)

if args.debug:
    fp_dataset_train = FloorplanGraphDataset(train_ids[-1:], hparams)
else:
    fp_dataset_train = FloorplanGraphDataset(train_ids, hparams)

fp_loader = torch.utils.data.DataLoader(
    fp_dataset_train,
    batch_size=1,
    shuffle=True,
    num_workers=hparams["n_cpu"],
    collate_fn=floorplan_collate_fn,
    pin_memory=False,
)


"""
  Loss, model, optimizer, tensorboard setup
"""
# loss functions (no GAN loss function technically)
if hparams["aux_loss"] == "MSE":
    aux_loss = torch.nn.MSELoss(reduction="none")
elif hparams["aux_loss"] == "L1":
    aux_loss = torch.nn.L1Loss(reduction="none")
elif hparams["aux_loss"] == "Huber":
    aux_loss = torch.nn.SmoothL1Loss(reduction="none")
else:
    raise Exception("Unknown auxiliary loss type")

# no black bits, NO BLACK BITS!!!!!!!
black_bits_loss = torch.nn.MSELoss()

# model definition
if hparams["image"]:
    from models_img import Discriminator, ResGen, compute_gradient_penalty

    generator = ResGen(hparams)
    discriminator = Discriminator(image_input=True, topo_input=hparams["topo_input"])
else:
    raise Exception("Image model only for now")
    from models import Discriminator, Generator, compute_gradient_penalty

    generator = Generator(image_input=False, topo_input=hparams["topo_input"])
    discriminator = Discriminator(image_input=False, topo_input=hparams["topo_input"])

with open(root_folder + "G_architecture.txt", "w") as f:
    print(generator, file=f)

with open(root_folder + "D_architecture.txt", "w") as f:
    print(discriminator, file=f)

if cuda:
    generator.cuda()
    discriminator.cuda()
    aux_loss.cuda()

# optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=hparams["g_lr"], betas=(hparams["b1"], hparams["b2"])
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=hparams["d_lr"], betas=(hparams["b1"], hparams["b2"])
)

# tensorboard summmary
writer = SummaryWriter(root_folder)


"""
  Resume if needed
"""
if args.resume:
    all_ckpt_files = sorted(glob.glob(root_folder + "*.pth"))
    assert len(all_ckpt_files)

    print("Resuming from %s" % all_ckpt_files[-1])
    last_ckpt = torch.load(all_ckpt_files[-1])

    generator.load_state_dict(last_ckpt["model_G"])
    discriminator.load_state_dict(last_ckpt["model_D"])
    optimizer_G.load_state_dict(last_ckpt["optimizer_G"])
    optimizer_D.load_state_dict(last_ckpt["optimizer_D"])
    epoch_start = last_ckpt["epoch"] + 1

else:
    epoch_start = 0


"""
  Training
"""
start = time.time()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

valid_label = torch.ones((64, 64)).type(Tensor)

optimizer_D.zero_grad()
optimizer_G.zero_grad()

for p in discriminator.parameters():
    p.requires_grad = False

for p in generator.parameters():
    p.requires_grad = False

d_batch_loss = 0
g_batch_loss = 0

for epoch_i in range(epoch_start, hparams["n_epochs"]):
    for batch_i, batch in enumerate(fp_loader):
        n_iter = epoch_i * len(fp_loader) + batch_i

        # Configure input
        given_masks = Variable(batch["given_masks"].type(Tensor)).unsqueeze(1)
        full_masks = Variable(batch["full_masks"].type(Tensor))
        ind_masks = Variable(batch["ind_masks"].type(Tensor)).unsqueeze(1)
        semantics = Variable(batch["semantics"].type(Tensor))
        graph_edges = batch["graph_edges"]
        ignore_masks = batch["ignore_masks"]
        nd_to_sample = batch["all_node_to_sample"]

        if hparams["image"]:
            given_imgs = Variable(batch["given_imgs"].type(Tensor))
        else:
            given_imgs = None

        if hparams["topo_input"]:
            topo_vecs = Variable(batch["topo_vecs"].type(Tensor))
        else:
            topo_vecs = None

        z_shape = [given_masks.shape[0], hparams["latent_dim"]]

        """
      Train Discriminator
    """
        if hparams["gan_loss"]:
            for p in discriminator.parameters():
                p.requires_grad = True

            # Generate random noise
            z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))

            # Generate a batch of images
            gen_masks, G_hidden = generator(
                z, given_masks, ind_masks, semantics, graph_edges, topo_vecs, given_imgs
            )

            # Real image scores
            real_validity, D_real_hidden = discriminator(
                full_masks,
                None,
                semantics,
                graph_edges,
                nd_to_sample,
                topo_vecs=topo_vecs,
                given_b=given_imgs,
            )

            # Fake image scores
            fake_validity, D_fake_hidden = discriminator(
                gen_masks.detach(),
                None,
                semantics.detach(),
                graph_edges.detach(),
                nd_to_sample.detach(),
                topo_vecs=topo_vecs,
                given_b=given_imgs,
            )

            # Measure discriminator's ability to classify real from generated samples
            gradient_penalty = compute_gradient_penalty(
                discriminator,
                full_masks.data,
                gen_masks.data,
                semantics.data,
                None,
                graph_edges.data,
                nd_to_sample.data,
                None,
                None,
                topo_vecs=topo_vecs,
                given_b=given_imgs,
            )

            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + hparams["lambda_gp"] * gradient_penalty
            )
            d_loss /= hparams["b_size"]

            # update discriminator
            d_loss.backward()
            d_batch_loss += d_loss.data

            # gradient accumulation
            if n_iter % hparams["b_size"] == 0:
                optimizer_D.step()
                optimizer_D.zero_grad()

            # Set grads off
            for p in discriminator.parameters():
                p.requires_grad = False

        else:
            real_validity = 0
            fake_validity = 0
            gradient_penalty = 0
            d_loss = 0

        writer.add_scalar("Discriminator/real", real_validity, n_iter)
        writer.add_scalar("Discriminator/fake", fake_validity, n_iter)
        writer.add_scalar("Discriminator/diff", real_validity - fake_validity, n_iter)
        writer.add_scalar("Discriminator/gp", gradient_penalty, n_iter)
        writer.add_scalar("Discriminator/d_loss", d_loss, n_iter)

        """
      Train Generator
    """
        for p in generator.parameters():
            p.requires_grad = True

        # generate a batch of images
        z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
        gen_masks, G_hidden = generator(
            z, given_masks, ind_masks, semantics, graph_edges, topo_vecs, given_imgs
        )

        # compute L1 loss
        loss_aux = aux_loss(gen_masks, full_masks)
        loss_aux[full_masks < gen_masks] /= hparams["neg_weight"]
        loss_aux = torch.abs(loss_aux)

        if hparams["ignore_gray"]:
            loss_aux[ignore_masks] = 0

        loss_aux = loss_aux.mean()

        # compute combined loss
        if hparams["gan_loss"]:
            # score fake images
            fake_validity, D_fake_hidden = discriminator(
                gen_masks,
                None,
                semantics,
                graph_edges,
                nd_to_sample,
                topo_vecs=topo_vecs,
                given_b=given_imgs,
            )
            g_loss = -torch.mean(fake_validity) + hparams["lambda_aux"] * loss_aux

        else:
            g_loss = loss_aux

        # Update generator
        g_loss /= hparams["b_size"]
        g_loss.backward()
        g_batch_loss += g_loss.data

        # Update optimizer
        if n_iter % hparams["b_size"] == 0:
            optimizer_G.step()
            optimizer_G.zero_grad()

            end = time.time()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [%.1f secs] [D: %.2f] [G: %.2f]"
                % (
                    epoch_i,
                    hparams["n_epochs"],
                    batch_i,
                    len(fp_loader),
                    end - start,
                    d_batch_loss,
                    g_batch_loss,
                )
            )
            start = time.time()

            d_batch_loss = 0
            g_batch_loss = 0

        writer.add_scalar("Generator/fake", fake_validity, n_iter)
        writer.add_scalar("Generator/loss_aux", loss_aux, n_iter)
        writer.add_scalar("Generator/g_loss", g_loss, n_iter)

        for p in generator.parameters():
            p.requires_grad = False

    # save model after each epoch
    torch.save(
        {
            "model_G": generator.state_dict(),
            "model_D": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "epoch": epoch_i,
        },
        root_folder + "%06d.pth" % epoch_i,
    )

    # visualizeSingleBatch(fp_loader_val, epoch)
