import os
import glob
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import measure, draw
from shapely.geometry import LineString, Polygon

import sem_rings
from find_corr import get_ins_sem, find_correspondence

random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))


GA_ROOT = ''
PREPROCESS_ROOT = GA_ROOT + 'preprocess/'
SPLITS_ROOT = GA_ROOT + 'splits/x_val/'


def rotate_to_new_instance(_ring):
  ring = _ring.copy()

  # rotate ring so we start on a new instance
  rot_idx = None

  for idx, (edge_a, edge_b) in enumerate(zip(ring[:-1], ring[1:])):
    _, _, a_id = edge_a
    _, _, b_id = edge_b

    if a_id == b_id:
      continue
    else:
      rot_idx = idx
      break

  if rot_idx != None:
    ring = ring[rot_idx+1:] + ring[:rot_idx+1]

  return ring


def get_separate_segments(ins_ring):
  neighbor_ids = set([x[2] for x in ins_ring])

  neighbor_segments = []
  for neighbor_id in neighbor_ids:
    neighbor_segment = [x for x in ins_ring if x[2] == neighbor_id]
    neighbor_segments.append(neighbor_segment)

  return neighbor_segments


def get_rays(ins_id, ins_full, ins_ring, all_ins_edges):
  poly_coords = [(edge[0][1], edge[0][0]) for (edge, _, _) in ins_ring]
  polygon = Polygon(poly_coords)
  centroid = np.array(polygon.centroid)

  neighbor_rays = []
  neighbor_segments = get_separate_segments(ins_ring)

  for neighbor_segment in neighbor_segments:
    neighbor_id = [x[2] for x in neighbor_segment]
    assert len(np.unique(neighbor_id)) == 1
    neighbor_id = neighbor_id[0]

    neighbor_sem = [x[1] for x in neighbor_segment]
    assert len(np.unique(neighbor_sem)) == 1
    neighbor_sem = neighbor_sem[0]

    if neighbor_sem == 0:
      continue

    if not len(all_ins_edges[neighbor_id]):
      continue

    poly_coords = [(a[1], a[0]) for (a,b) in all_ins_edges[neighbor_id]]
    polygon = Polygon(poly_coords)
    assert polygon.is_valid
    mid_pt = np.array(polygon.centroid)

    # boundary_pixels = []
    # for ((a,b), _, _) in neighbor_segment:
    #   rr, cc = draw.line(a[0], a[1], b[0], b[1])
    #   boundary_pixels.extend(list(zip(rr, cc)))
    # mid_pt = np.array(boundary_pixels).mean(axis=0)[::-1]

    # coords = [x[0][0] for x in neighbor_segment]
    # coords.append(neighbor_segment[-1][0][1])
    # line = LineString(coords)
    # mid_pt = np.array(line.interpolate(0.5, normalized=True))[::-1]

    neighbor_rays.append((neighbor_id, neighbor_sem, mid_pt))

  return centroid, neighbor_rays


# https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
def get_angle(va, vb):
  va = va / np.linalg.norm(va)
  vb = vb / np.linalg.norm(vb)
  dot_product = np.dot(va, vb)
  angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

  return np.degrees(angle)


def match_rays(pred_centroid, pred_rays, gt_centroid, gt_rays,
               handles_only=False, threshold=15):
  pred_left = []
  gt_left = []

  # find all the instances we need to find matches for
  for pred_i, (pred_id, pred_sem, pred_end) in enumerate(pred_rays):
    if handles_only:
      if pred_sem == 7:
        pred_left.append(pred_i)
    else:
      pred_left.append(pred_i)

  for gt_i, (gt_id, gt_sem, gt_end) in enumerate(gt_rays):
    if handles_only:
      if gt_sem == 7:
        gt_left.append(gt_i)
    else:
      gt_left.append(gt_i)

  # potential matches first, then we greedily find best ones
  potential_matches = []
  for pred_i in pred_left:
    for gt_i in gt_left:
      pred_id, pred_sem, pred_end = pred_rays[pred_i]
      gt_id, gt_sem, gt_end = gt_rays[gt_i]

      if pred_sem != gt_sem:
        continue

      pred_end = (pred_end - pred_centroid).copy()
      gt_end = (gt_end - gt_centroid).copy()
      angle = get_angle(pred_end, gt_end)

      if angle > threshold:
        continue
      else:
        potential_matches.append((angle, (pred_i, gt_i)))

  # greedily find best matches
  matches = []
  potential_matches = sorted(potential_matches)

  for angle, (pred_i, gt_i) in potential_matches:
    if (not len(pred_left)) or (not len(gt_left)):
      break

    if (pred_i not in pred_left) or (gt_i not in gt_left):
      continue

    if (pred_i in pred_left) and (gt_i in gt_left):
      matches.append((pred_i, gt_i))
      pred_left.remove(pred_i)
      gt_left.remove(gt_i)

  # if we have any leftovers, then that means topology is incorrect
  if len(pred_left) or len(gt_left):
    return False, matches
  else:
    return True, matches


def get_bbox(ins_id, ins_full, crop_margin=16, min_side_len=128):
  ins_mask = (ins_full == ins_id)

  ii, jj = np.nonzero(ins_mask)
  side_len = max(ii.max() - ii.min() + 2 * crop_margin,
                 jj.max() - jj.min() + 2 * crop_margin,
                 min_side_len)
  center_i = (ii.max() + ii.min()) // 2
  center_j = (jj.max() + jj.min()) // 2

  mini = max(center_i - side_len // 2, 0)
  minj = max(center_j - side_len // 2, 0)
  mini = min(mini, ins_full.shape[0] - side_len)
  minj = min(minj, ins_full.shape[1] - side_len)

  maxi = min(mini + side_len, ins_full.shape[0])
  maxj = min(minj + side_len, ins_full.shape[1])

  assert (mini >= 0) and (minj >= 0)
  assert (maxi <= ins_full.shape[0]) and (maxj <= ins_full.shape[1])

  return [mini, minj, maxi, maxj]


print('Door eval for r2v')

semantic_fs = glob.glob(GA_ROOT + 'r2v_results/all/*_result_line.npy')
fp_ids = [x.split('/')[-1][:14] for x in semantic_fs]
assert len(fp_ids) == 200

all_total = 0
all_matched = 0

handle_total = 0
handle_matched = 0

for idx, fp_id in enumerate(fp_ids):
  print('%d / %d' % (idx, len(fp_ids)))

  gt_sem_full = np.load(PREPROCESS_ROOT + 'semantic/%s.npy' % fp_id)
  pred_sem_full = np.load(GA_ROOT + 'r2v_results/all/%s_result_line.npy' % fp_id)

  gt_ins_full = measure.label(gt_sem_full, background=-1, connectivity=1)
  pred_ins_full = measure.label(pred_sem_full, background=-1, connectivity=1)

  gt_ins_edges = sem_rings.get_instance_edge_mapping(gt_ins_full, gt_sem_full, pad=False)
  gt_sem_rings = sem_rings.get_sem_rings(gt_ins_edges, gt_ins_full, gt_sem_full)
  pred_ins_edges = sem_rings.get_instance_edge_mapping(pred_ins_full, pred_sem_full, pad=False)
  pred_sem_rings = sem_rings.get_sem_rings(pred_ins_edges, pred_ins_full, pred_sem_full)

  pred_to_gt = find_correspondence(pred_ins_full, pred_sem_full, gt_ins_full, gt_sem_full)

  mapped_gt_ids = []
  for pred_id, gt_id in pred_to_gt.items():
    if get_ins_sem(pred_id, pred_ins_full, pred_sem_full) not in [3,4,5]:
      continue

    if gt_id < 0:
      continue

    gt_centroid, gt_rays = get_rays(gt_id,
                                    gt_ins_full,
                                    gt_sem_rings[gt_id][1],
                                    gt_ins_edges)
    pred_centroid, pred_rays = get_rays(pred_id,
                                        pred_ins_full,
                                        pred_sem_rings[pred_id][1],
                                        pred_ins_edges)

    _all_matched, all_matches = match_rays(pred_centroid, pred_rays,
                                          gt_centroid, gt_rays,
                                          handles_only=False)
    _handle_matched, handle_matches = match_rays(pred_centroid, pred_rays,
                                                gt_centroid, gt_rays,
                                                handles_only=True)

    all_total += 1
    handle_total += 1

    if _all_matched:
      all_matched += 1
    if _handle_matched:
      handle_matched += 1

    # if not rays_matched:
    if False:
      fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True,
                                     dpi=150, figsize=(32,18))

      ax1.imshow(pred_sem_full / 7., cmap='nipy_spectral',
                 vmin=0., vmax=1., interpolation='nearest')
      ax2.imshow(gt_sem_full / 7., cmap='nipy_spectral',
                 vmin=0., vmax=1., interpolation='nearest')

      pred_matched = [x[0] for x in matches]
      gt_matched = [x[1] for x in matches]

      for pred_i, (_, pred_sem, pred_end) in enumerate(pred_rays):
        # if pred_sem != 7:
        #   continue

        xx = [pred_centroid[0], pred_end[0]]
        yy = [pred_centroid[1], pred_end[1]]

        if pred_i in pred_matched:
          ax1.plot(xx, yy, '-c')
        else:
          ax1.plot(xx, yy, '-y')

      for gt_i, (_, gt_sem, gt_end) in enumerate(gt_rays):
        # if gt_sem != 7:
        #   continue

        xx = [gt_centroid[0], gt_end[0]]
        yy = [gt_centroid[1], gt_end[1]]

        if gt_i in gt_matched:
          ax2.plot(xx, yy, '-c')
        else:
          ax2.plot(xx, yy, '-y')

      ax1.set_axis_off()
      ax2.set_axis_off()

      # mini, minj, maxi, maxj = get_bbox(pred_id, pred_ins_full)
      # ax1.set_xlim(minj, maxj)
      # ax1.set_ylim(maxi, mini)

      plt.suptitle('Matched: %s' % str(rays_matched))
      plt.tight_layout()
      save_f = './door_vis/%s/%s_%d.png' % (args.input_folder, fp_id, pred_id)
      os.makedirs(os.path.dirname(save_f), exist_ok=True)
      plt.savefig(save_f, bbox_inches='tight', pad_inches=0.1)
      plt.close()

print('all: %d / %d = %.3f' \
        % (all_matched, all_total, all_matched / all_total))
print('handle: %d / %d = %.3f' \
        % (handle_matched, handle_total, handle_matched / handle_total))

with open('./topo_r2v.txt', 'w') as f:
  f.write('all: %d / %d = %.3f\n' \
            % (all_matched, all_total, all_matched / all_total))
  f.write('handle: %d / %d = %.3f\n' \
            % (handle_matched, handle_total, handle_matched / handle_total))
