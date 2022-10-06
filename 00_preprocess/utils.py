import os
import shutil
import argparse

import numpy as np

from tqdm import tqdm
from ruamel.yaml import YAML
from skimage.transform import resize
from skimage.morphology import erosion, square


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

  parser.add_argument('--single', dest='single',
                      action='store_true', default=False)

  parser.add_argument('--debug', dest='debug',
                      action='store_true', default=False)

  args = parser.parse_args()

  return args


def get_floorplan_bbox(label_semantic, label_instance, border_size=32):
  # find the bounding box of the label annotation, plus a small padding
  ii_semantic, jj_semantic = np.nonzero(label_semantic)
  ii_instance, jj_instance = np.nonzero(label_instance)

  # NOTE I used to check the ii's and jj's, but in reality I think we just
  # need to check that the bboxes computed from both match
  # assert (ii_semantic == ii_instance).all()
  # assert (jj_semantic == jj_instance).all()

  assert ii_semantic.min() == ii_instance.min()
  assert ii_semantic.max() == ii_instance.max()
  assert jj_semantic.min() == jj_instance.min()
  assert jj_semantic.max() == jj_instance.max()

  ii = ii_semantic
  jj = jj_semantic

  # leave a small border around the crop
  mini = max(ii.min() - border_size, 0)
  minj = max(jj.min() - border_size, 0)
  maxi = min(ii.max() + border_size, label_semantic.shape[0])
  maxj = min(jj.max() + border_size, label_semantic.shape[1])

  bbox = [int(mini), int(minj), int(maxi), int(maxj)]

  return bbox

  # after padding and reshaping, we would like to know the bounding box of the
  # floorplan, so here we make a white image of the same shape as the
  # floorplan, pad it and reshape it with the rest of the data, and then find
  # the bounding box of the white mask
  white_mask = np.full(image_crop.shape, fill_value=1.0)

  # pad the crops to squares
  img_h, img_w = image_crop.shape

  if img_h > img_w:
    pad_before = (img_h - img_w) // 2
    pad_after = img_h - pad_before - img_w
    pad_width = np.array([[0, 0], [pad_before, pad_after]])

  else:
    pad_before = (img_w - img_h) // 2
    pad_after = img_w - pad_before - img_h
    pad_width = np.array([[pad_before, pad_after], [0, 0]])

  image_padded = np.pad(image_crop, pad_width, mode='constant')
  label_semantic_padded = np.pad(label_semantic_crop,
                                 pad_width,
                                 mode='constant',
                                 constant_values=-1)
  label_instance_padded = np.pad(label_instance_crop,
                                 pad_width,
                                 mode='constant',
                                 constant_values=-1)
  white_mask_padded = np.pad(white_mask, pad_width, mode='constant')

  # reshape the padded crops to the network input shape
  image_padded = resize(image_padded, [crop_size, crop_size])
  label_semantic_padded = resize(label_semantic_padded,
                                 [crop_size, crop_size],
                                 order=0,
                                 preserve_range=True,
                                 anti_aliasing=False)
  label_instance_padded = resize(label_instance_padded,
                                 [crop_size, crop_size],
                                 order=0,
                                 preserve_range=True,
                                 anti_aliasing=False)
  white_mask_padded = resize(white_mask_padded,
                             [crop_size, crop_size],
                             order=0,
                             preserve_range=True,
                             anti_aliasing=False)

  # find the bounding box of the floorplan after padding and reshape
  assert (np.unique(white_mask_padded) == [0.0, 1.0]).all()
  ii_white_mask, jj_white_mask = np.nonzero(white_mask_padded)

  bbox_mini = ii_white_mask.min()
  bbox_minj = jj_white_mask.min()
  bbox_maxi = ii_white_mask.max()
  bbox_maxj = jj_white_mask.max()

  bbox = [bbox_mini, bbox_minj, bbox_maxi, bbox_maxj]

  return image_padded, label_semantic_padded, label_instance_padded, bbox


def generate_boundary_mask(all_instance_mask, k=10):
  boundary_mask = np.zeros_like(all_instance_mask, dtype=np.float32)

  for instance_id in np.unique(all_instance_mask):
    if instance_id == 0:
      continue

    instance_mask = (all_instance_mask == instance_id)

    level = 1.0
    while instance_mask.any():
      boundary_mask[instance_mask] = level
      instance_mask = erosion(instance_mask, square(k))
      level += 1

  return boundary_mask
