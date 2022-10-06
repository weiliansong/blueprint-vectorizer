import os
import glob
import json
import shutil

from multiprocessing import Pool

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import measure
from shapely.geometry import Point, Polygon
from shapely.algorithms.polylabel import polylabel

import utils
import paths
import sem_rings


random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))

class_map = {
  0: 'BG',
  1: 'OW',
  2: 'IW',
  3: 'WD',
  4: 'DR',
  5: 'FR',
  6: 'RM',
  7: 'SY',
}


def restore_dict(flat_dict):
  # it should be a 1D array
  assert len(flat_dict.shape) == 1

  hist_dict = {
    'top': flat_dict[:8],
    'right': flat_dict[8:16],
    'bottom': flat_dict[16:24],
    'left': flat_dict[24:32],

    'top_symbol': flat_dict[32:48],
    'right_symbol': flat_dict[48:64],
    'bottom_symbol': flat_dict[64:80],
    'left_symbol': flat_dict[80:96],

    'top_symbols': [flat_dict[32:40], flat_dict[40:48]],
    'right_symbols': [flat_dict[48:56], flat_dict[56:64]],
    'bottom_symbols': [flat_dict[64:72], flat_dict[72:80]],
    'left_symbols': [flat_dict[80:88], flat_dict[88:96]]
  }

  return hist_dict


def get_label(hist_dict):
  # label = [
  #   hist_dict['top'], hist_dict['right'],
  #   hist_dict['bottom'], hist_dict['left'],
  #   hist_dict['top_symbol'], hist_dict['right_symbol'],
  #   hist_dict['bottom_symbol'], hist_dict['left_symbol'],
  # ]

  # return np.concatenate(label)

  label = []

  for key in ['top_symbol', 'right_symbol', 'bottom_symbol', 'left_symbol']:
    label.append(hist_dict[key][:8].max())
    label.append(hist_dict[key][8:].max())

  return np.array(label)


def label_title(label):
  title_str = ''
  hist_dict = restore_dict(label)

  for key in ['top', 'right', 'bottom', 'left']:
    side_hist = hist_dict[key]
    side_str = ', '.join([class_map[a] for (a,b) in enumerate(side_hist) if b])
    title_str += '%s: %s\n' % (key, side_str)

  return title_str


def topology_title(hist_dict):
  title_str = ''

  for key in ['top', 'right', 'bottom', 'left']:
    side_hist = hist_dict[key]
    sym_hist = hist_dict[key + '_symbol']

    side_str = ', '.join([class_map[a] for (a,b) in enumerate(side_hist) if b])
    sym_1_str = ', '.join([class_map[a] for (a,b) in enumerate(sym_hist[:8]) if b])
    sym_2_str = ', '.join([class_map[a] for (a,b) in enumerate(sym_hist[8:]) if b])

    title_str += '%s: %s | S1: %s | S2: %s\n' % (key, side_str, sym_1_str, sym_2_str)

  return title_str


def vis_bad_pair(pred_id, p_ins_full, p_sem_full, g_sem_full):
  fig, [ax1, ax2] = plt.subplots(ncols=2)

  p_ins_crop, p_sem_crop, bbox = get_crop(p_ins_full, p_sem_full, pred_id)

  mini, minj, maxi, maxj = bbox
  g_sem_crop = g_sem_full[mini:maxi, minj:maxj]

  ax1.imshow(p_sem_crop / 7., cmap='nipy_spectral',
             vmin=0., vmax=1., interpolation='nearest')
  ax2.imshow(g_sem_crop / 7., cmap='nipy_spectral',
             vmin=0., vmax=1., interpolation='nearest')

  ax1.imshow((p_ins_crop == pred_id), cmap='gray', alpha=0.7)

  ax1.set_axis_off()
  ax2.set_axis_off()
  plt.tight_layout()
  plt.show()
  plt.close()


def vis_pred_gt_pair(pred_id, gt_id, p_ins_full, p_sem_full, g_ins_full,
                     g_sem_full, pred_label, gt_label, fname):
  fig, [ax1, ax2] = plt.subplots(ncols=2)

  p_ins_crop, p_sem_crop, _ = get_crop(p_ins_full, p_sem_full, pred_id)
  g_ins_crop, g_sem_crop, _ = get_crop(g_ins_full, g_sem_full, gt_id)

  ax1.imshow(p_sem_crop / 7., cmap='nipy_spectral',
             vmin=0., vmax=1., interpolation='nearest')
  ax2.imshow(g_sem_crop / 7., cmap='nipy_spectral',
             vmin=0., vmax=1., interpolation='nearest')

  ax1.imshow((p_ins_crop == pred_id), cmap='gray', alpha=0.7)
  ax2.imshow((g_ins_crop == gt_id), cmap='gray', alpha=0.7)

  ax1.set_title('Pred\n' + topology_title(pred_label))
  ax2.set_title('GT\n' + topology_title(gt_label))

  ax1.set_axis_off()
  ax2.set_axis_off()
  plt.tight_layout()
  plt.savefig(fname, dpi=200, bbox_inches='tight', pad_inches=0.1)
  plt.close()


def one_hot_embedding(labels, num_classes=8):
  """Embedding labels to one-hot form.

  Args:
    labels: (LongTensor) class labels, sized [N,].
    num_classes: (int) number of classes.

  Returns:
    (tensor) encoded labels, sized [N, #classes].
  """
  y = torch.eye(num_classes)
  return y[labels]


# handles 2D array
def to_onehot(indices, num_classes=None):
  if not num_classes:
    num_classes = indices.max()+1

  # remember the 2D shape and flatten it
  (h,w) = indices.shape
  indices = indices.flatten()

  onehot = np.zeros((indices.size, num_classes), dtype=np.float32)
  onehot[np.arange(indices.size), indices] = 1

  onehot = np.reshape(onehot, [h, w, num_classes])
  onehot = np.transpose(onehot, [2,0,1])

  return onehot


def get_semantic(ins_full, sem_full, ins_id):
  ins_mask = (ins_full == ins_id)
  ins_sem = sem_full[ins_mask]
  assert len(np.unique(ins_sem)) == 1
  return ins_sem[0]


def get_crop(ins_full, sem_full, ins_id, crop_margin=16, min_side_len=256):
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

  ins_crop = ins_full[mini:maxi, minj:maxj]
  sem_crop = sem_full[mini:maxi, minj:maxj]

  return ins_crop, sem_crop, (mini, minj, maxi, maxj)


def get_crops(fp_id, crop_margin=16, min_side_len=256):
  sem_full = np.load(paths.PREPROCESS_ROOT + 'semantic/%s.npy' % fp_id)
  ins_full = measure.label(sem_full, background=-1, connectivity=1)
  sem_full[sem_full == 24] = 7

  for ins_id in np.unique(ins_full):
    ins_mask = (ins_full == ins_id)
    ins_sem = sem_full[ins_mask]
    assert len(np.unique(ins_sem)) == 1
    ins_sem = ins_sem[0]

    if ins_sem not in [3,4,5]:
      continue

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

    ins_crop = ins_full[mini:maxi, minj:maxj]
    sem_crop = sem_full[mini:maxi, minj:maxj]

    yield ins_crop, sem_crop, ins_id


def vis_ring(all_ins_mask, all_sem_mask, target_id, instance_ring):
  ins_mask = (all_ins_mask == target_id)

  margin = 50
  ii, jj = np.nonzero(ins_mask)

  mini = max(ii.min() - margin, 0)
  minj = max(jj.min() - margin, 0)
  maxi = min(ii.max() + margin, all_ins_mask.shape[0])
  maxj = min(jj.max() + margin, all_ins_mask.shape[1])

  ins_crop = all_ins_mask[mini:maxi, minj:maxj]
  sem_crop = all_sem_mask[mini:maxi, minj:maxj]

  plt.figure()
  plt.imshow(sem_crop / 7., cmap='nipy_spectral',
             interpolation='nearest', vmin=0., vmax=1.)

  # plot the perimeter points
  xx, yy = [], []
  for edge, _, _ in instance_ring:
    xx.append(edge[0][1]-0.5-minj)
    xx.append(edge[1][1]-0.5-minj)
    yy.append(edge[0][0]-0.5-mini)
    yy.append(edge[1][0]-0.5-mini)

  plt.plot(xx, yy, '-o', linewidth=1, markersize=3)

  plt.axis('off')
  plt.tight_layout()
  plt.show()
  plt.close()


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


def remove_collinear(_ring):
  ring = _ring.copy()

  # rotate ring so we start on a corner
  rot_idx = None

  for idx, (edge_a, edge_b) in enumerate(zip(ring[:-1], ring[1:])):
    ((a_i, a_j), (b_i, b_j)), _, _ = edge_a
    ((b_i, b_j), (c_i, c_j)), _, _ = edge_b

    if (a_i == b_i == c_i) or (a_j == b_j == c_j):
      continue
    else:
      rot_idx = idx
      break

  ring = ring[rot_idx:] + ring[:rot_idx]

  # check consecutive pairs of edges
  ring.append(ring[0])

  noncollinear_pts = []
  noncollinear_idxes = []

  for idx, (edge_a, edge_b) in enumerate(zip(ring[:-1], ring[1:])):
    ((a_i, a_j), (b_i, b_j)), _, _ = edge_a
    ((b_i, b_j), (c_i, c_j)), _, _ = edge_b

    if (a_i == b_i == c_i) or (a_j == b_j == c_j):
      continue
    else:
      noncollinear_pts.append((b_i, b_j))
      noncollinear_idxes.append((idx + rot_idx) % len(_ring))

  if noncollinear_pts[0] != noncollinear_pts[-1]:
    noncollinear_pts.append(noncollinear_pts[0])
    noncollinear_idxes.append(noncollinear_idxes[0])

  return noncollinear_pts, noncollinear_idxes


def flatten_dict(hist_dict):
  order = ['top', 'right', 'bottom', 'left']
  flatten = [hist_dict[side] for side in order]
  return np.concatenate(flatten, axis=0)


def get_edge_length(edge):
  (x1,y1), (x2,y2) = edge
  assert (x1 == x2) or (y1 == y2)
  return max(abs(x2 - x1), abs(y2 - y1))


def get_segment_length(segment):
  total_length = 0.

  for (edge, _, _) in segment:
    total_length += get_edge_length(edge)

  return total_length


# this is using the centroid method
def get_four_sides_1(instance_ring):
  # DEBUG vis_ring(all_ins_mask, all_sem_mask, target_id, instance_ring)
  noncollinear_pts, noncollinear_idxes = remove_collinear(instance_ring)

  # subtract centroid and find the four corners
  poly_coords = [(pt[1], pt[0]) for pt in noncollinear_pts]
  polygon = Polygon(poly_coords)
  centroid = polygon.centroid

  if not polygon.contains(centroid):
    centroid = polylabel(polygon)

  normalized_pts = [(pt[0]-centroid.x, pt[1]-centroid.y) for pt in poly_coords]
  normalized_pts = np.array(normalized_pts)

  tl_idx = None
  tr_idx = None
  bl_idx = None
  br_idx = None

  tl_max = 0
  tr_max = 0
  bl_max = 0
  br_max = 0

  areas = abs(normalized_pts[:,0] * normalized_pts[:,1])
  for pt_idx, pt, area in zip(noncollinear_idxes, normalized_pts, areas):
    if (pt[0] > 0) and (pt[1] > 0) and (area > tr_max):
      tr_idx = (pt_idx+1) % len(instance_ring)
      tr_max = area

    if (pt[0] > 0) and (pt[1] <= 0) and (area > br_max):
      br_idx = (pt_idx+1) % len(instance_ring)
      br_max = area

    if (pt[0] <= 0) and (pt[1] <= 0) and (area > bl_max):
      bl_idx = (pt_idx+1) % len(instance_ring)
      bl_max = area

    if (pt[0] <= 0) and (pt[1] > 0) and (area > tl_max):
      tl_idx = (pt_idx+1) % len(instance_ring)
      tl_max = area

  if (tl_idx == None) or (tr_idx == None) or (bl_idx == None) or (br_idx == None):
    assert 'Check this instance for four-way hist'

  # get the four sides to the polygon
  if tl_idx < tr_idx:
    top_edge = instance_ring[tl_idx:tr_idx]
  else:
    top_edge = instance_ring[tl_idx:] + instance_ring[:tr_idx]

  if tr_idx < br_idx:
    right_edge = instance_ring[tr_idx:br_idx]
  else:
    right_edge = instance_ring[tr_idx:] + instance_ring[:br_idx]

  if br_idx < bl_idx:
    bottom_edge = instance_ring[br_idx:bl_idx]
  else:
    bottom_edge = instance_ring[br_idx:] + instance_ring[:bl_idx]

  if bl_idx < tl_idx:
    left_edge = instance_ring[bl_idx:tl_idx]
  else:
    left_edge = instance_ring[bl_idx:] + instance_ring[:tl_idx]

  edges = {
    'top': top_edge,
    'right': right_edge,
    'bottom': bottom_edge,
    'left': left_edge
  }
  return edges


# this is using the bbox method
def get_four_sides_2(instance_ring):
  if sem_rings.is_ccw(instance_ring):
    instance_ring = instance_ring[::-1]

  poly_coords = [(edge[0][1], edge[0][0]) for (edge, _, _) in instance_ring]
  polygon = Polygon(poly_coords)
  minx, miny, maxx, maxy = polygon.bounds

  # find the indices of the points closest to the bounding box corners
  def find_closest(source, targets):
    closest_idx = -1
    closest_dist = float('inf')
    targets = [Point(x) for x in targets]

    for idx, target in enumerate(targets):
      if source.distance(target) < closest_dist:
        closest_idx = idx
        closest_dist = source.distance(target)

    return closest_idx

  targets = polygon.exterior.coords
  tl_idx = find_closest(Point(minx, maxy), poly_coords)
  tr_idx = find_closest(Point(maxx, maxy), poly_coords)
  br_idx = find_closest(Point(maxx, miny), poly_coords)
  bl_idx = find_closest(Point(minx, miny), poly_coords)

  # get the four sides to the polygon
  if tl_idx < tr_idx:
    top_edge = instance_ring[tl_idx:tr_idx]
  else:
    top_edge = instance_ring[tl_idx:] + instance_ring[:tr_idx]

  if tr_idx < br_idx:
    right_edge = instance_ring[tr_idx:br_idx]
  else:
    right_edge = instance_ring[tr_idx:] + instance_ring[:br_idx]

  if br_idx < bl_idx:
    bottom_edge = instance_ring[br_idx:bl_idx]
  else:
    bottom_edge = instance_ring[br_idx:] + instance_ring[:bl_idx]

  if bl_idx < tl_idx:
    left_edge = instance_ring[bl_idx:tl_idx]
  else:
    left_edge = instance_ring[bl_idx:] + instance_ring[:tl_idx]

  edges = {
    'top': top_edge,
    'right': right_edge,
    'bottom': bottom_edge,
    'left': left_edge,

    'tl': poly_coords[tl_idx],
    'tr': poly_coords[tr_idx],
    'br': poly_coords[br_idx],
    'bl': poly_coords[bl_idx],
    'bbox': [minx, miny, maxx, maxy]
  }
  return edges


def get_symbol_hist(sem_ring):
  hist = np.zeros(8, dtype=np.int64)

  for (_, sem, _) in sem_ring:
    hist[sem] = 1.

  return hist


def get_four_way_hist(all_ins_mask, all_sem_mask, all_sem_rings, target_id):
  semantic, instance_ring = all_sem_rings[target_id]
  assert semantic in [3,4,5]

  edges = get_four_sides_2(instance_ring)
  top_edge = edges['top']
  right_edge = edges['right']
  bottom_edge = edges['bottom']
  left_edge = edges['left']

  # split the semantic ring by instance
  prev_id = -1
  prev_segment = []
  neighbor_segments = []

  for (n_edge, n_sem, n_id) in rotate_to_new_instance(instance_ring):
    if prev_id < 0:
      prev_id = n_id
      prev_segment.append((n_edge, n_sem, n_id))

    else:
      if prev_id == n_id:
        prev_segment.append((n_edge, n_sem, n_id))
      else:
        neighbor_segments.append(prev_segment)
        prev_id = n_id
        prev_segment = [(n_edge, n_sem, n_id),]

  neighbor_segments.append(prev_segment)  # we would have missed the last one

  # generate four histograms
  top_hist = np.zeros(8, dtype=np.int64)
  right_hist = np.zeros(8, dtype=np.int64)
  bottom_hist = np.zeros(8, dtype=np.int64)
  left_hist = np.zeros(8, dtype=np.int64)

  top_symbol_hist = []
  right_symbol_hist = []
  bottom_symbol_hist = []
  left_symbol_hist = []

  for neighbor_segment in neighbor_segments:
    top_coverage = 0.
    right_coverage = 0.
    bottom_coverage = 0.
    left_coverage = 0.

    for neighbor_edge in neighbor_segment:
      # ignore edges that are on the perimeter
      if neighbor_edge[1] < 0:
        raise Exception

      dist = get_edge_length(neighbor_edge[0])

      if neighbor_edge in top_edge:
        top_coverage += dist
      elif neighbor_edge in right_edge:
        right_coverage += dist
      elif neighbor_edge in bottom_edge:
        bottom_coverage += dist
      elif neighbor_edge in left_edge:
        left_coverage += dist
      else:
        raise Exception('Edge not in any of the four sides?')

    # normalize the coverage by the total length of that side
    top_coverage /= get_segment_length(top_edge)
    right_coverage /= get_segment_length(right_edge)
    bottom_coverage /= get_segment_length(bottom_edge)
    left_coverage /= get_segment_length(left_edge)

    neighbor_id = [x[2] for x in neighbor_segment]
    assert len(np.unique(neighbor_id)) == 1
    neighbor_id = neighbor_id[0]

    neighbor_semantic = [x[1] for x in neighbor_segment]
    assert len(np.unique(neighbor_semantic)) == 1
    neighbor_semantic = neighbor_semantic[0]

    # if we cover at least 25% of that edge, we also toggle it
    # also keep track of symbol topology
    added = False
    max_coverage = max(top_coverage, right_coverage,
                       bottom_coverage, left_coverage)

    if (max_coverage == top_coverage) or (top_coverage >= 0.25):
      added = True
      top_hist[neighbor_semantic] = 1

      if neighbor_semantic == 7:
        _, symbol_sem_ring = all_sem_rings[neighbor_id]
        top_symbol_hist.append(get_symbol_hist(symbol_sem_ring))

    if (max_coverage == right_coverage) or (right_coverage >= 0.25):
      added = True
      right_hist[neighbor_semantic] = 1

      if neighbor_semantic == 7:
        _, symbol_sem_ring = all_sem_rings[neighbor_id]
        right_symbol_hist.append(get_symbol_hist(symbol_sem_ring))

    if (max_coverage == bottom_coverage) or (bottom_coverage >= 0.25):
      added = True
      bottom_hist[neighbor_semantic] = 1

      if neighbor_semantic == 7:
        _, symbol_sem_ring = all_sem_rings[neighbor_id]
        bottom_symbol_hist.append(get_symbol_hist(symbol_sem_ring))

    if (max_coverage == left_coverage) or (left_coverage >= 0.25):
      added = True
      left_hist[neighbor_semantic] = 1

      if neighbor_semantic == 7:
        _, symbol_sem_ring = all_sem_rings[neighbor_id]
        left_symbol_hist.append(get_symbol_hist(symbol_sem_ring))

    # debug vis
    # if semantic in [3,4,5]:
    if False:
      print('T: %f' % top_coverage)
      print('R: %f' % right_coverage)
      print('B: %f' % bottom_coverage)
      print('L: %f' % left_coverage)

      ins_mask = (all_ins_mask == target_id)

      margin = 50
      ii, jj = np.nonzero(ins_mask)

      mini = max(ii.min() - margin, 0)
      minj = max(jj.min() - margin, 0)
      maxi = min(ii.max() + margin, all_ins_mask.shape[0])
      maxj = min(jj.max() + margin, all_ins_mask.shape[1])

      ins_crop = all_ins_mask[mini:maxi, minj:maxj]
      sem_crop = all_sem_mask[mini:maxi, minj:maxj]

      plt.figure()
      plt.imshow(sem_crop / 7., cmap='nipy_spectral',
                 interpolation='nearest', vmin=0., vmax=1.)

      # plot the four edges
      for side in neighbor_segments:
        xx, yy = [], []
        for edge, _, _ in side:
          xx.append(edge[0][1]-0.5-minj)
          xx.append(edge[1][1]-0.5-minj)
          yy.append(edge[0][0]-0.5-mini)
          yy.append(edge[1][0]-0.5-mini)

        plt.plot(xx, yy, '-o', linewidth=3, markersize=7)

      # plot the four edges
      xx, yy = [], []
      for edge, _, _ in neighbor_segment:
        xx.append(edge[0][1]-0.5-minj)
        xx.append(edge[1][1]-0.5-minj)
        yy.append(edge[0][0]-0.5-mini)
        yy.append(edge[1][0]-0.5-mini)

      plt.plot(xx, yy, '*', linewidth=3, markersize=7)

      plt.axis('off')
      plt.tight_layout()
      plt.show()
      plt.close()

    assert added

  # each side can have a max of 2 symbols, otherwise skip this object
  if (len(top_symbol_hist) > 2) or (len(right_symbol_hist) > 2) \
      or (len(bottom_symbol_hist) > 2) or (len(left_symbol_hist) > 2):
    return {}

  def process_symbol_hist(symbol_hist):
    blank_hist = np.zeros(8, dtype=np.int64)

    if len(symbol_hist) == 0:
      return np.concatenate([blank_hist, blank_hist])
    elif len(symbol_hist) == 1:
      return np.concatenate([symbol_hist[0], blank_hist])
    elif len(symbol_hist) == 2:
      return np.concatenate(symbol_hist)
    else:
      raise Exception('Something slipped through!')

  top_symbol_hist = process_symbol_hist(top_symbol_hist)
  right_symbol_hist = process_symbol_hist(right_symbol_hist)
  bottom_symbol_hist = process_symbol_hist(bottom_symbol_hist)
  left_symbol_hist = process_symbol_hist(left_symbol_hist)

  # due to how origin is at top-left, top is actually in reality bottom
  tmp_hist = top_hist
  top_hist = bottom_hist
  bottom_hist = tmp_hist

  tmp_symbol_hist = top_symbol_hist
  top_symbol_hist = bottom_symbol_hist
  bottom_symbol_hist = tmp_symbol_hist

  # pack everything into one dict
  hist_dict = {
    'top': top_hist,
    'right': right_hist,
    'bottom': bottom_hist,
    'left': left_hist,

    'top_edge': top_edge,
    'right_edge': right_edge,
    'bottom_edge': bottom_edge,
    'left_edge': left_edge,

    'top_symbol': top_symbol_hist,
    'right_symbol': right_symbol_hist,
    'bottom_symbol': bottom_symbol_hist,
    'left_symbol': left_symbol_hist,

    'tl': edges['tl'],
    'tr': edges['tr'],
    'br': edges['br'],
    'bl': edges['bl'],
    'bbox': edges['bbox']
  }

  if False:
    ins_mask = (all_ins_mask == target_id)

    margin = 50
    ii, jj = np.nonzero(ins_mask)

    mini = max(ii.min() - margin, 0)
    minj = max(jj.min() - margin, 0)
    maxi = min(ii.max() + margin, all_ins_mask.shape[0])
    maxj = min(jj.max() + margin, all_ins_mask.shape[1])

    ins_crop = all_ins_mask[mini:maxi, minj:maxj]
    sem_crop = all_sem_mask[mini:maxi, minj:maxj]

    plt.figure()
    plt.imshow(sem_crop / 7., cmap='nipy_spectral',
               interpolation='nearest', vmin=0., vmax=1.)

    # plot the four edges
    for side in [top_edge, right_edge, bottom_edge, left_edge]:
      xx, yy = [], []
      for edge, _, _ in side:
        xx.append(edge[0][1]-0.5-minj)
        xx.append(edge[1][1]-0.5-minj)
        yy.append(edge[0][0]-0.5-mini)
        yy.append(edge[1][0]-0.5-mini)

      plt.plot(xx, yy, '-o', linewidth=3, markersize=7)

    # plot the four-way histograms
    plt.title(topology_title(hist_dict))

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

  return hist_dict


if __name__ == '__main__':
  fp_id = ''
  instance_f = paths.INSTANCE_ROOT + '%s.npy' % fp_id
  semantic_f = paths.SEMANTIC_ROOT + '%s.npy' % fp_id

  all_instance_mask = np.load(instance_f, allow_pickle=False)
  all_semantic_mask = np.load(semantic_f, allow_pickle=False)

  # woops semantic class is kind of inconsistent...
  all_semantic_mask[all_semantic_mask == 24] = 7

  # we need the semantic rings
  instance_to_edges = sem_rings.get_instance_edge_mapping(all_instance_mask,
                                                          all_semantic_mask)

  all_sem_rings = sem_rings.get_sem_rings(instance_to_edges,
                                          all_instance_mask,
                                          all_semantic_mask)

  # two instances of interest
  target_id = 91

  if not target_id:
    fig, [ax1, ax2] = plt.subplots(ncols=2)

    ax1.imshow(all_instance_mask, cmap=random_cmap)
    ax2.imshow(all_semantic_mask, cmap='nipy_spectral')

    ax1.set_axis_off()
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()
    plt.close()

    target_id = int(input('Enter target instance ID: '))

  # print('Target pair: (%d, %d)' % (id_A, id_B))

  # NOTE testing
  haha = get_four_way_hist(all_instance_mask,
                           all_semantic_mask,
                           all_sem_rings,
                           target_id)
