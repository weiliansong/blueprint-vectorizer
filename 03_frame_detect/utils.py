import os
import shutil
import argparse

from itertools import groupby

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import measure
from ruamel.yaml import YAML
from torchvision import transforms

import paths

interp_mode = transforms.InterpolationMode


def ensure_dir(path_name):
  if not os.path.exists(os.path.dirname(path_name)):
    os.makedirs(os.path.dirname(path_name), exist_ok=True)


def remove_dir(path_name):
  if os.path.exists(os.path.dirname(path_name)):
    shutil.rmtree(os.path.dirname(path_name))


def parse_yaml(config_path):
  yaml = YAML(typ='safe')
  config = yaml.load(open(config_path, 'r'))

  return config


def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('--restart', dest='restart',
                      action='store_true', default=False)

  parser.add_argument('--resume', dest='resume',
                      action='store_true', default=False)

  parser.add_argument('--single', dest='single',
                      action='store_true', default=False)

  parser.add_argument('--debug', dest='debug',
                      action='store_true', default=False)

  parser.add_argument('--overfit', dest='overfit',
                      action='store_true', default=False)

  parser.add_argument('--fp_id', dest='fp_id',
                      action='store', type=str, default='')

  parser.add_argument('--hparam_f', dest='hparam_f',
                      action='store', type=str, default='./base.yaml')

  args = parser.parse_args()

  return args


# by default it performs NN resizing
def resize(image, size, interpolation=interp_mode.NEAREST):
  if len(image.shape) == 2:
    unsqueezed = True
    image = torch.tensor(image).unsqueeze(0)
  elif len(image.shape) == 3:
    unsqueezed = False
    image = torch.tensor(image).permute([2,0,1])
  else:
    raise Exception('Unknown image format')

  resizer = transforms.Resize(size, interpolation=interpolation)
  image = resizer(image)

  if unsqueezed:
    return image.squeeze().numpy()
  else:
    return image.permute([1,2,0]).numpy()


def get_cropped_segmentations(fp_id):
  instance_f = paths.INSTANCE_ROOT + '%s.npy' % fp_id
  semantic_f = paths.SEMANTIC_ROOT + '%s.npy' % fp_id

  all_instance_mask = np.load(instance_f, allow_pickle=False)
  all_semantic_mask = np.load(semantic_f, allow_pickle=False)
  assert all_instance_mask.shape == all_semantic_mask.shape

  # crop the empty borders
  ii, jj = np.nonzero(semantic_full)

  mini = max(ii.min() - 32, 0)
  minj = max(jj.min() - 32, 0)
  maxi = min(ii.max() + 32, semantic_full.shape[0])
  maxj = min(jj.max() + 32, semantic_full.shape[1])

  all_instance_mask = all_instance_mask[mini:maxi, minj:maxj]
  all_semantic_mask = all_semantic_mask[mini:maxi, minj:maxj]


def save_gt_instance_visualization(fp_id, instance_id, save_root, plot_title):
  instance_f = paths.INSTANCE_ROOT + '%s.npy' % fp_id
  semantic_f = paths.SEMANTIC_ROOT + '%s.npy' % fp_id

  all_instance_mask = np.load(instance_f, allow_pickle=False)
  all_semantic_mask = np.load(semantic_f, allow_pickle=False)
  assert all_instance_mask.shape == all_semantic_mask.shape

  instance_mask = (all_instance_mask == instance_id)

  margin = 50
  ii, jj = np.nonzero(instance_mask)

  mini = max(ii.min() - margin, 0)
  minj = max(jj.min() - margin, 0)
  maxi = min(ii.max() + margin, all_instance_mask.shape[0])
  maxj = min(jj.max() + margin, all_instance_mask.shape[1])

  instance_crop = all_instance_mask[mini:maxi, minj:maxj]
  semantic_crop = all_semantic_mask[mini:maxi, minj:maxj]

  plt.figure()
  plt.imshow(semantic_crop / 24., cmap='nipy_spectral',
             interpolation='nearest', vmin=0., vmax=1.)
  plt.axis('off')
  plt.title(plot_title)
  plt.tight_layout()
  plt.savefig(save_root + '%s_%d.png' % (fp_id, instance_id))
  plt.close()


def get_instance_semantic(instance_id, all_instance_mask, all_semantic_mask):
  instance_mask = (all_instance_mask == instance_id)
  semantic_mask = all_semantic_mask[instance_mask]
  unique, counts = np.unique(semantic_mask, return_counts=True)
  instance_sem = unique[np.argmax(counts)]

  return instance_sem


def circularly_identical(list1, list2):
  _list1 = ['|%d|' % x for x in list1]
  _list2 = ['|%d|' % x for x in list2]
  return (' '.join(_list1) in ' '.join(_list2 * 2)) or \
            (' '.join(_list1[::-1]) in ' '.join(_list2 * 2))


def find_circular_key(query, keys):
  identical_key = None

  if keys:
    for key in keys:
      if circularly_identical(query, key) and (len(query) == len(key)):
        assert not identical_key
        identical_key = key

  return identical_key


# since input is a circular list, we also check the two ends as well
def remove_consecutive_duplicates(ring):
  if not len(ring):
    return ring

  if len(np.unique(ring)) == 1:
    return [ring[0],]

  _ring = [x[0] for x in groupby(ring)]

  if _ring[0] == _ring[-1]:
    return _ring[:-1]
  else:
    return _ring


# remove small bits in semantic floorplan
def remove_holes(all_instance_mask, all_semantic_mask):
  cleaned_semantic_mask = all_semantic_mask.copy()

  for instance_id in np.unique(all_instance_mask):
    all_neighbor_sems = semantic_adj_list.get_sem_adj_list(instance_id,
                                                           all_instance_mask,
                                                           all_semantic_mask)

    if (len(all_neighbor_sems) == 1) and (all_neighbor_sems[0][1] == 6):
      instance_mask = (all_instance_mask == instance_id)
      cleaned_semantic_mask[instance_mask] = all_neighbor_sems[0][1]

      if False:
        fig, [ax1, ax2] = plt.subplots(ncols=2)

        ax1.imshow(all_semantic_mask, cmap='nipy_spectral')
        ax2.imshow(cleaned_semantic_mask, cmap='nipy_spectral')
        ax2.imshow(instance_mask, cmap='hot', alpha=0.8)

        ax1.set_axis_off()
        ax2.set_axis_off()

        plt.tight_layout()
        plt.show()
        plt.close()

  return cleaned_semantic_mask
