import os
import numpy as np
import pickle
from new_utils import *
from skimage import measure

from find_corr import get_ins_sem, find_correspondence


def crop_object(ins_id, ins_full, sem_full, crop_margin=32, min_side_len=128):
  ins_mask = (ins_full == ins_id)

  ii, jj = np.nonzero(ins_mask)
  mini = max(ii.min() - crop_margin, 0)
  minj = max(jj.min() - crop_margin, 0)
  maxi = min(ii.max() + crop_margin, ins_full.shape[0])
  maxj = min(jj.max() + crop_margin, ins_full.shape[1])

  assert (mini >= 0) and (minj >= 0)
  assert (maxi <= ins_full.shape[0]) and (maxj <= ins_full.shape[1])

  ins_crop = ins_full[mini:maxi, minj:maxj]
  sem_crop = sem_full[mini:maxi, minj:maxj]

  return ins_crop, sem_crop, [mini, minj, maxi, maxj]


class Metric():
    def calc(self, gt_data, conv_data, thresh=8.0, iou_thresh=0.7):
        ### compute corners precision/recall
        gts = gt_data['corners']
        dets = conv_data['corners']

        per_sample_corner_tp = 0.0
        per_sample_corner_fp = 0.0
        per_sample_corner_length = gts.shape[0]
        found = [False] * gts.shape[0]
        c_det_annot = {}

        # for each corner detection
        for i, det in enumerate(dets):
            # get closest gt
            near_gt = [0, 999999.0, (0.0, 0.0)]
            for k, gt in enumerate(gts):
                dist = np.linalg.norm(gt-det)
                if dist < near_gt[1]:
                    near_gt = [k, dist, gt]
            if near_gt[1] <= thresh and not found[near_gt[0]]:
                per_sample_corner_tp += 1.0
                found[near_gt[0]] = True
                c_det_annot[i] = near_gt[0]
            else:
                per_sample_corner_fp += 1.0

        per_corner_score = {
            'recall': per_sample_corner_tp / gts.shape[0],
            'precision': per_sample_corner_tp/(per_sample_corner_tp+per_sample_corner_fp+1e-8)
        }

        ### compute edges precision/recall
        per_sample_edge_tp = 0.0
        per_sample_edge_fp = 0.0
        edge_corner_annots = gt_data['edges']
        per_sample_edge_length = edge_corner_annots.shape[0]

        false_edge_ids = []
        match_gt_ids = set()

        for l, e_det in enumerate(conv_data['edges']):
            c1, c2 = e_det

            # check if corners are mapped
            if (c1 not in c_det_annot.keys()) or (c2 not in c_det_annot.keys()):
                per_sample_edge_fp += 1.0
                false_edge_ids.append(l)
                continue
            # check hit
            c1_prime = c_det_annot[c1]
            c2_prime = c_det_annot[c2]
            is_hit = False

            for k, e_annot in enumerate(edge_corner_annots):
                c3, c4 = e_annot
                if ((c1_prime==c3) and (c2_prime==c4)) or ((c1_prime==c4) and (c2_prime==c3)):
                    is_hit = True
                    match_gt_ids.add(k)
                    break

            # hit
            if is_hit:
                per_sample_edge_tp += 1.0
            else:
                per_sample_edge_fp += 1.0
                false_edge_ids.append(l)

        per_edge_score = {
            'recall': per_sample_edge_tp/edge_corner_annots.shape[0],
            'precision': per_sample_edge_tp/(per_sample_edge_tp+per_sample_edge_fp+1e-8)
        }

        ## compute region
        conv_sem_full = conv_data['segmentation']
        gt_sem_full = gt_data['segmentation']
        conv_ins_full = measure.label(conv_sem_full, background=-1, connectivity=1)
        gt_ins_full = measure.label(gt_sem_full, background=-1, connectivity=1)

        conv_to_gt = find_correspondence(conv_ins_full, conv_sem_full,
                                         gt_ins_full, gt_sem_full)

        per_sample_region_tp = 0.0
        per_sample_region_fp = 0.0
        per_sample_region_length = len(np.unique(gt_ins_full))

        per_sem_region_tp = {}
        per_sem_region_fp = {}
        per_sem_region_length = {}

        for sem in range(8):
          per_sem_region_tp[sem] = 0.0
          per_sem_region_fp[sem] = 0.0
          per_sem_region_length[sem] = 0.0

        mapped_gt_ids = []

        gt_mappings = {}
        pred_mappings = {}
        for gt_id in np.unique(gt_ins_full):
          gt_mappings[gt_id] = -1
          gt_sem = get_ins_sem(gt_id, gt_ins_full, gt_sem_full)
          per_sem_region_length[gt_sem] += 1.0

        for (conv_id, gt_id) in conv_to_gt.items():
          matched = False
          pred_mappings[conv_id] = -1

          conv_sem = get_ins_sem(conv_id, conv_ins_full, conv_sem_full)

          if gt_id < 0:
            per_sample_region_fp += 1.0
            per_sem_region_fp[conv_sem] += 1.0
          elif gt_id in mapped_gt_ids:
            per_sample_region_fp += 1.0
            per_sem_region_fp[conv_sem] += 1.0
          else:
            conv_mask = (conv_ins_full == conv_id)
            gt_mask = (gt_ins_full == gt_id)

            region_iou = (conv_mask & gt_mask).sum() / (conv_mask | gt_mask).sum()
            if region_iou > 0.0:
              per_sample_region_tp += 1.0
              per_sem_region_tp[conv_sem] += 1.0
              mapped_gt_ids.append(gt_id)
              matched = True
              gt_mappings[gt_id] = conv_id
              pred_mappings[conv_id] = gt_id
            else:
              per_sample_region_fp += 1.0
              per_sem_region_fp[conv_sem] += 1.0

          if False:
            if get_ins_sem(conv_id, conv_ins, conv_sem) == 0:
              continue

            conv_ins_crop, conv_sem_crop, bbox = crop_object(conv_id,
                                                             conv_ins,
                                                             conv_sem)
            mini, minj, maxi, maxj = bbox
            gt_ins_crop = gt_ins[mini:maxi, minj:maxj]
            gt_sem_crop = gt_sem[mini:maxi, minj:maxj]

            fig, [ax1, ax2] = plt.subplots(ncols=2)

            ax1.imshow(conv_sem_crop / 7., cmap='nipy_spectral',
                       vmin=0., vmax=1., interpolation='nearest')
            ax1.imshow(conv_ins_crop == conv_id, cmap='gray', alpha=0.5)

            ax2.imshow(gt_sem_crop / 7., cmap='nipy_spectral',
                       vmin=0., vmax=1., interpolation='nearest')
            ax2.imshow(gt_ins_crop == gt_id, cmap='gray', alpha=0.5)

            ax1.set_axis_off()
            ax2.set_axis_off()

            plt.suptitle('Matched: %s' % str(matched))
            plt.tight_layout()
            plt.show()
            plt.close()

        return {
            'corner_tp': per_sample_corner_tp,
            'corner_fp': per_sample_corner_fp,
            'corner_length': per_sample_corner_length,

            'edge_tp': per_sample_edge_tp,
            'edge_fp': per_sample_edge_fp,
            'edge_length': per_sample_edge_length,

            'region_tp': per_sample_region_tp,
            'region_fp': per_sample_region_fp,
            'region_length': per_sample_region_length,

            'region_sem_tp': per_sem_region_tp,
            'region_sem_fp': per_sem_region_fp,
            'region_sem_length': per_sem_region_length,

            'gt_mappings': gt_mappings,
            'pred_mappings': pred_mappings
        }


def get_recall_and_precision(tp, fp, length):
    recall = tp / (length + 1e-8)
    precision = tp / (tp+fp+1e-8)
    return recall, precision
