import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable

import utils


def convert_full_masks(full_masks, semantics):
  full_masks = full_masks.numpy()
  semantics = semantics.numpy()

  order = np.argsort(full_masks.sum(axis=(1,2)))[::-1]
  full_masks = full_masks[order]
  semantics = semantics[order]

  _, h, w = full_masks.shape
  sem_crop = np.zeros([h,w], dtype=np.int64)

  for ins_id, (full_mask, semantic) in enumerate(zip(full_masks, semantics)):
    ins_mask = (full_mask > 0)
    sem_crop[ins_mask] = semantic.argmax()

  return sem_crop


def convert_gen_masks(gen_masks, semantics, bbox=None, fix_bits=False):
  if type(gen_masks) != np.ndarray:
    gen_masks = gen_masks.cpu().detach().numpy()
  if type(semantics) != np.ndarray:
    semantics = semantics.cpu().detach().numpy()

  _, h, w = gen_masks.shape
  sem_crop = np.zeros([h,w], dtype=np.int64)

  for gen_mask, semantic in zip(gen_masks, semantics):
    ins_mask = (gen_mask > 0)
    sem_crop[ins_mask] = semantic.argmax() + 1

  if fix_bits:
    while 0 in sem_crop:
      ii, jj = np.nonzero(sem_crop == 0)

      for (i,j) in zip(ii, jj):
        mini = max(i-1, 0)
        minj = max(j-1, 0)
        maxi = min(i+2, sem_crop.shape[0])
        maxj = min(j+2, sem_crop.shape[1])

        unique, count = np.unique(sem_crop[mini:maxi, minj:maxj],
                                  return_counts=True)

        for sem in unique[np.argsort(count)]:
          if sem != 0:
            sem_crop[i,j] = sem
            break

    assert 0 not in sem_crop

  if bbox:
    mini, minj, maxi, maxj = bbox
    new_h = maxi - mini
    new_w = maxj - minj
    assert new_h == new_w
    sem_crop = utils.resize(sem_crop, [new_h, new_w])

  return sem_crop - 1


def generator_forward(generator, batch, opt, cuda):
  Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

  # Configure input
  given_masks = generator_inputs['given_masks'].type(Tensor)
  ind_masks = generator_inputs['ind_masks'].type(Tensor)
  semantics = generator_inputs['semantics'].type(Tensor)
  graph_edges = generator_inputs['graph_edges']
  given_imgs = generator_inputs['given_imgs'].type(Tensor)

  z_shape = [given_masks.shape[0], opt.latent_dim]

  debug_sample = {
    'given_masks': given_masks,
    'ignore_masks': generator_inputs['ignore_masks'],
    'ind_masks': generator_inputs['ind_masks'],
    'given_imgs': generator_inputs['given_imgs'],
    'full_masks': generator_inputs['full_masks'],
    'semantics': generator_inputs['semantics']
  }

  # add channel to indicate given nodes
  given_masks = given_masks.unsqueeze(1)
  # ind_masks = ind_masks.unsqueeze(1)

  # if opt.image:
  #   given_masks = torch.cat([given_masks, ind_masks, given_imgs], 1)
  # else:
  #   given_masks = torch.cat([given_masks, ind_masks], 1)

  step_vis = []
  with torch.no_grad():
    z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
    gen_masks, _ = generator(z, given_masks, semantics, graph_edges, given_imgs)
    step_vis.append(convert_gen_masks(gen_masks, semantics))

    debug_sample['gen_masks'] = gen_masks.detach()
    # utils.vis_input(0, debug_sample, extra_fname=str(0))

    prev_pred = None
    patience = opt.refine_patience

    for step_i in range(opt.refine_num_steps):
      given_masks = gen_masks.unsqueeze(1).detach()
      # new_inds_masks = torch.zeros_like(gen_masks)

      debug_sample = {
        'given_masks': given_masks.squeeze(),
        'ignore_masks': generator_inputs['ignore_masks'],
        'ind_masks': generator_inputs['ind_masks'],
        'given_imgs': generator_inputs['given_imgs'],
        'full_masks': generator_inputs['full_masks'],
        'semantics': generator_inputs['semantics']
      }

      # if opt.image:
      #   given_masks = torch.cat([gen_masks, ind_masks, given_imgs], 1)
      # else:
      #   given_masks = torch.cat([gen_masks, ind_masks], 1)

      z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
      gen_masks, _ = generator(z, given_masks, semantics, graph_edges, given_imgs)

      debug_sample['gen_masks'] = gen_masks.detach()
      # utils.vis_input(0, debug_sample, extra_fname=str(step_i+1))

      curr_pred = convert_gen_masks(gen_masks, semantics)
      step_vis.append(curr_pred)

      if (prev_pred == curr_pred).all():
        if patience:
          patience -= 1
        else:
          break

      else:
        prev_pred = curr_pred
        patience = opt.refine_patience

  return gen_masks, step_vis
