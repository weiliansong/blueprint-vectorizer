import os
import json

import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from tqdm import tqdm
from skimage import measure
from skimage.morphology import skeletonize
from scipy.ndimage import binary_erosion, binary_dilation
from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset

import utils
import data_utils

from refinery_2 import Refinery
from find_corr import find_correspondence

interp_mode = transforms.InterpolationMode


class FloorplanGraphDataset(Dataset):

  def __init__(self, split_ids, hparams, hide_prob=[0,1], shrink_prob=[0,1]):
    super().__init__()
    self.hparams = hparams
    self.hide_prob = hide_prob
    self.shrink_prob = shrink_prob

    # get the corresponding floorplan IDs for the split we're using
    fp_ids = []

    for split_id in split_ids:
      split_json = utils.SPLITS_ROOT + 'ids_%d.json' % split_id
      with open(split_json, 'r') as f:
        fp_ids.extend(json.load(f))

    print('Caching examples...')
    self.examples = []
    self.all_img_full = {}
    self.all_gt_ins_full = {}
    self.all_gt_sem_full = {}
    self.all_pred_ins_full = {}
    self.all_pred_sem_full = {}
    self.all_pred_to_gt_corr = {}

    for fp_id in tqdm(fp_ids):
      img_full = np.array(Image.open(utils.PREPROCESS_ROOT + 'fp_img/%s.jpg' % fp_id), dtype=np.float32) / 255.
      gt_sem_full = np.load(utils.PREPROCESS_ROOT + 'semantic/%s.npy' % fp_id)
      gt_ins_full = measure.label(gt_sem_full, background=-1, connectivity=1)
      # pred_sem_full = np.load(utils.PREPROCESS_ROOT + 'semantic_pred/%s.npy' % fp_id)
      # pred_ins_full = measure.label(pred_sem_full, background=-1, connectivity=1)

      # pred_sem_full[pred_sem_full == 24] = 7
      assert (gt_sem_full.min() >= 0) and (gt_sem_full.max() <= 7)

      self.all_img_full[fp_id] = img_full
      self.all_gt_ins_full[fp_id] = gt_ins_full
      self.all_gt_sem_full[fp_id] = gt_sem_full
      # self.all_pred_ins_full[fp_id] = pred_ins_full
      # self.all_pred_sem_full[fp_id] = pred_sem_full

      if hparams['sliding_bboxes']:
        raise Exception('Not doing this')

        sliding_window = hparams['sliding_window']
        sliding_stride = hparams['sliding_stride']

        assert gt_ins_full.shape == pred_ins_full.shape
        h,w = gt_ins_full.shape

        for i in range(0, h // sliding_stride):
          mini = i * sliding_stride
          maxi = mini + sliding_window

          if mini >= h:
            continue
          if maxi > h:
            maxi = h
            mini = h - sliding_window

          for j in range(0, w // sliding_stride):
            minj = j * sliding_stride
            maxj = minj + sliding_window

            if minj >= w:
              continue
            if maxj > w:
              maxj = w
              minj = w - sliding_window

            ins_crop = utils.resize(gt_ins_full[mini:maxi, minj:maxj], [64,64])
            if len(np.unique(ins_crop)) <= 1:
              continue

            if hparams['use_gt']:
              example = {
                'fp_id': fp_id,
                'input_type': 'gt',
                'bbox': [mini, minj, maxi, maxj],
              }
              self.examples.append(example)

            if hparams['use_pred']:
              example = {
                'fp_id': fp_id,
                'input_type': 'pred',
                'bbox': [mini, minj, maxi, maxj],
              }
              self.examples.append(example)

      if hparams['object_bboxes']:
        refiner = Refinery(fp_id, gt_ins_full, gt_sem_full, img_full,
                           None, None, pred=False)

        for example in refiner.get_examples():
          self.examples.append(example)


  def get_crop_bboxes(self, ins_crop, target_id):
    bboxes = []
    assert ins_crop.shape[0] == ins_crop.shape[1]

    if ins_crop.shape[0] < 200:
      bboxes.append([0, 0, ins_crop.shape[0], ins_crop.shape[1]])

    else:
      seeds = (ins_crop == target_id)

      h,w = ins_crop.shape
      valid_area = np.zeros_like(ins_crop, dtype=bool)
      valid_area[64:h-64, 64:w-64] = True

      if np.logical_and(seeds, valid_area).max():
        seeds = np.logical_and(seeds, valid_area)
      seeds = np.array(list(zip(*np.nonzero(seeds))))

      if len(seeds) < 25:
        seeds = seeds[:1]
      else:
        idxes = np.random.choice(range(len(seeds)), size=10, replace=False)
        seeds = seeds[idxes]

      side_len = 128
      for (center_i, center_j) in seeds:
        mini = max(center_i - side_len // 2, 0)
        minj = max(center_j - side_len // 2, 0)
        mini = min(mini, ins_crop.shape[0] - side_len)
        minj = min(minj, ins_crop.shape[1] - side_len)

        maxi = min(mini + side_len, ins_crop.shape[0])
        maxj = min(minj + side_len, ins_crop.shape[1])

        assert (mini >= 0) and (minj >= 0)
        assert (maxi <= ins_crop.shape[0]) and (maxj <= ins_crop.shape[1])

        bboxes.append([mini, minj, maxi, maxj])

    return bboxes


  def __len__(self):
    return len(self.examples)


  def __getitem__(self, index):
    return data_utils.get_network_inputs(self.examples[index],
                                         hide_prob=self.hide_prob,
                                         shrink_prob=self.shrink_prob)


def floorplan_collate_fn(batch):
  all_sample = {
    'given_masks': [],
    'ind_masks': [],
    'semantics': [],
    'graph_edges': [],
    'given_imgs': [],
    'full_masks': [],
    'ignore_masks': [],
    'all_node_to_sample': [],
    'all_edge_to_sample': [],
  }

  node_offset = 0
  for i, sample in enumerate(batch):
    O, T = sample['semantics'].size(0), sample['graph_edges'].size(0)

    for key in ['given_masks', 'semantics', 'given_imgs',
                'full_masks', 'ignore_masks', 'ind_masks']:
      all_sample[key].append(sample[key])

    graph_edges = sample['graph_edges'].clone()
    assert len(graph_edges)
    graph_edges[:, 0] += node_offset
    graph_edges[:, 2] += node_offset
    all_sample['graph_edges'].append(graph_edges)

    all_sample['all_node_to_sample'].append(torch.LongTensor(O).fill_(i))
    all_sample['all_edge_to_sample'].append(torch.LongTensor(T).fill_(i))
    node_offset += O

  for key in all_sample.keys():
    all_sample[key] = torch.cat(all_sample[key])

  return all_sample


def main():
  args = utils.parse_arguments()
  hparams = utils.parse_config(args.hparam_f)
  my_dataset = FloorplanGraphDataset(range(1), hparams)

  for i, sample in enumerate(tqdm(my_dataset)):
    sample['gen_masks'] = sample['given_masks']
    full_vis = utils.vis_input(sample)
    full_vis = Image.fromarray(full_vis.astype('uint8'))
    full_vis.save('vis/%03d.png' % i)
    continue

    ins_crop = np.zeros([64,64], dtype=np.int32)
    sem_crop = np.zeros([64,64], dtype=np.float32)
    centroids = {}

    for ins_id, (given_mask, full_mask, semantic) in enumerate(zip(sample['given_masks'],
                                                                   sample['full_masks'],
                                                                   sample['semantics'])):
      full_mask = (full_mask.numpy() > 0.)
      given_mask = (given_mask.numpy() > 0.)
      semantic = semantic.numpy().argmax()

      assert not ins_crop[full_mask].sum()
      ins_crop[given_mask] = ins_id
      sem_crop[given_mask] = semantic

      props = measure.regionprops(full_mask.astype(int))
      assert len(props) == 1
      centroids[ins_id] = props[0]['centroid']

    plt.imshow(sem_crop / 7., cmap='nipy_spectral',
               vmin=0., vmax=1., interpolation='nearest')
    plt.imshow(ins_crop, alpha=0)

    # for (a, c, b) in sample['graph_edges'].numpy():
    #   if c > 0:
    #     a_y, a_x = centroids[a]
    #     b_y, b_x = centroids[b]

    #     plt.plot([a_x, b_x], [a_y, b_y], '--oc')

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

    continue

    sample['gen_masks'] = sample['given_masks']
    utils.vis_input('./vis/%d.png' % i, sample)


if __name__ == '__main__':
  main()
