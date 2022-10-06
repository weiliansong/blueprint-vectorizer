import os
import glob
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from skimage import measure
from skimage.morphology import binary_dilation, square


def get_ins_sem(ins_id, ins_full, sem_full):
  ins_mask = (ins_full == ins_id)
  sem_mask = sem_full[ins_mask]
  unique, counts = np.unique(sem_mask, return_counts=True)
  ins_sem = unique[np.argmax(counts)]

  return ins_sem


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
    pred_instance_mask = binary_dilation(pred_instance_mask, square(5))
    labels, counts = np.unique(all_gt_instance_mask[pred_instance_mask],
                               return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    labels = labels[sort_idx]
    counts = counts[sort_idx]

    gt_id = -1

    for label in labels:
      gt_sem = get_ins_sem(label, all_gt_instance_mask, all_gt_semantic_mask)

      if gt_sem == pred_sem:
        gt_id = label.tolist()
        break

    # save this mapping from prediction to GT
    # tolist() to have an int instead of np.int, this is for JSON compatibility
    pred_to_gt_corr[pred_id.tolist()] = gt_id

    if False:
      fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

      ax1.imshow(all_pred_semantic_mask, cmap='nipy_spectral')
      ax1.imshow(pred_instance_mask, cmap='hot', alpha=0.4)

      ax2.imshow(all_gt_semantic_mask, cmap='nipy_spectral')
      ax2.imshow(pred_instance_mask, cmap='hot', alpha=0.4)

      # ii, jj = np.nonzero(pred_instance_mask)
      # ax1.set_xlim(jj.min()+10, jj.max()-10)
      # ax1.set_ylim(ii.min()+10, ii.max()-10)

      ax1.set_axis_off()
      ax2.set_axis_off()

      plt.title('Matched: %s' % str(gt_id >= 0))
      plt.tight_layout()
      plt.show()
      plt.close()

  return pred_to_gt_corr
