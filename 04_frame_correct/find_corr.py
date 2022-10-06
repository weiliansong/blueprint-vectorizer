import os
import glob
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from skimage import measure

import data_utils


def find_correspondence(all_pred_instance_mask,
                        all_pred_semantic_mask,
                        all_gt_instance_mask,
                        all_gt_semantic_mask):
  pred_to_gt_corr = {}

  for pred_id in np.unique(all_pred_instance_mask):
    pred_instance_mask = (all_pred_instance_mask == pred_id)

    pred_sem = np.unique(all_pred_semantic_mask[pred_instance_mask])
    assert len(pred_sem) == 1
    pred_sem = pred_sem[0]

    # vote on the instance label
    labels, counts = np.unique(all_gt_instance_mask[pred_instance_mask],
                               return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    labels = labels[sort_idx]
    counts = counts[sort_idx]

    gt_id = -1

    for label in labels:
      gt_sem = data_utils.get_ins_sem(label,
                                      all_gt_instance_mask,
                                      all_gt_semantic_mask)

      if gt_sem == pred_sem:
        gt_id = label.tolist()
        break

    # save this mapping from prediction to GT
    # tolist() to have an int instead of np.int, this is for JSON compatibility
    pred_to_gt_corr[pred_id.tolist()] = gt_id

  return pred_to_gt_corr
