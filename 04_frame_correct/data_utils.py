import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from skimage.morphology import skeletonize
from torchvision import transforms
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

import utils
import sem_rings

interp_mode = transforms.InterpolationMode


def compute_num_errors(label, pred):
  assert (max(pred) <= 1.) and (min(pred) >= 0.)
  assert (max(label) <= 1.) and (min(label) >= 0.)

  pred[pred > 0.5] = 1.
  pred[pred <= 0.5] = 0.

  label = label.astype(int)
  pred = pred.astype(int)

  errors = (label != pred)

  return np.sum(errors)


def restore_dict(flat_dict):
  # it should be a 1D array
  assert len(flat_dict.shape) == 1

  hist_dict = {
    'top': flat_dict[:2],
    'right': flat_dict[2:4],
    'bottom': flat_dict[4:6],
    'left': flat_dict[6:8],

    # 'top_symbols': [flat_dict[32:40], flat_dict[40:48]],
    # 'right_symbols': [flat_dict[48:56], flat_dict[56:64]],
    # 'bottom_symbols': [flat_dict[64:72], flat_dict[72:80]],
    # 'left_symbols': [flat_dict[80:88], flat_dict[88:96]]
  }

  return hist_dict


def crop_object(ins_id, ins_full, sem_full, crop_margin=16, min_side_len=128):
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

  return ins_crop, sem_crop, [mini, minj, maxi, maxj]


def get_ins_sem(ins_id, ins_full, sem_full):
  ins_mask = (ins_full == ins_id)
  sem_mask = sem_full[ins_mask]
  unique, counts = np.unique(sem_mask, return_counts=True)
  ins_sem = unique[np.argmax(counts)]

  return ins_sem


def get_new_id(target_mask, target_sem, ins_crop, sem_crop, dilate=True):
  if dilate:
    struct = generate_binary_structure(2, 2)
    target_mask = binary_dilation(target_mask, structure=struct)

  unique, count = np.unique(ins_crop[target_mask], return_counts=True)

  return_id = -1
  for ins_id in unique[np.argsort(count)[::-1]]:
    if get_ins_sem(ins_id, ins_crop, sem_crop) == target_sem:
      return_id = ins_id
      break

  return return_id


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


def distort_crop(ins_crop, sem_crop, prob):
  nodes, edges = sem_rings.get_graph(ins_crop, pad=True)
  edges = MultiLineString(edges)

  new_edges = []
  for edge in edges:
    if edge.length >= 4:
      num_vert = np.random.choice([2,3,4]) + 2
      distances = sorted(np.random.uniform(size=num_vert))
      distances[0] = 0.0
      distances[-1] = 1.0
      new_edge = LineString([edge.interpolate(dist, normalized=True)
                              for dist in distances])
    else:
      new_edge = edge

    new_coords = np.array(new_edge).round().astype(int)
    for pair in zip(new_coords[:-1], new_coords[1:]):
      if (pair[0] != pair[1]).any():
        new_edges.append(pair)

  # sem_rings.plot_edges(ins_crop, new_edges)

  (h,w) = ins_crop.shape
  new_ins_crop = ins_crop.copy()
  new_sem_crop = sem_crop.copy()

  for (a,b) in new_edges:
    ai, aj = int(a[0]), int(a[1])
    bi, bj = int(b[0]), int(b[1])

    # this is a horizontal edge
    if ai == bi:
      n_mini = max(ai - 1, 0)
      n_maxi = min(ai + 1, h)
      n_minj = max(min(aj, bj), 0)
      n_maxj = min(max(aj, bj), w)

    # this is a vertical edge
    elif aj == bj:
      n_mini = max(min(ai, bi), 0)
      n_maxi = min(max(ai, bi), h)
      n_minj = max(aj - 1, 0)
      n_maxj = min(aj + 1, w)

    else:
      raise Exception('non-manhattan edge')

    neighbor_ids = np.unique(ins_crop[n_mini:n_maxi, n_minj:n_maxj])

    # don't touch edges near boundary
    if len(neighbor_ids) == 1:
      continue

    # pick random (direction and) magnitude
    magnitude = np.random.choice([-1, 0, 1], p=[prob, 1-2*prob, prob])

    if not magnitude:
      continue

    # shift edge without changing topology
    if ai == bi:
      c_mini = max(min(ai, ai - magnitude), 0)
      c_maxi = min(max(ai, ai - magnitude), h)

      p_mini = max(min(ai, ai + magnitude), 0)
      p_maxi = min(max(ai, ai + magnitude), h)

      c_minj = p_minj = max(min(aj, bj), 0)
      c_maxj = p_maxj = min(max(aj, bj), w)

    elif aj == bj:
      c_mini = p_mini = max(min(ai, bi), 0)
      c_maxi = p_maxi = min(max(ai, bi), h)

      c_minj = max(min(aj, aj - magnitude), 0)
      c_maxj = min(max(aj, aj - magnitude), w)

      p_minj = max(min(aj, aj + magnitude), 0)
      p_maxj = min(max(aj, aj + magnitude), w)

    else:
      raise Exception('non-manhattan edge')

    if False:
      plt.imshow(ins_crop, cmap=utils.random_cmap)

      p_xx = np.array([p_minj, p_maxj, p_maxj, p_minj, p_minj]) - 0.5
      p_yy = np.array([p_mini, p_mini, p_maxi, p_maxi, p_mini]) - 0.5
      plt.plot(p_xx, p_yy, '-ob')

      c_xx = np.array([c_minj, c_maxj, c_maxj, c_minj, c_minj]) - 0.5
      c_yy = np.array([c_mini, c_mini, c_maxi, c_maxi, c_mini]) - 0.5
      plt.plot(c_xx, c_yy, '-or')

      plt.show()
      plt.close()

    assert len(np.unique(ins_crop[c_mini:c_maxi, c_minj:c_maxj])) == 1
    assert len(np.unique(sem_crop[c_mini:c_maxi, c_minj:c_maxj])) == 1
    copy_ins = np.unique(ins_crop[c_mini:c_maxi, c_minj:c_maxj])[0]
    copy_sem = np.unique(sem_crop[c_mini:c_maxi, c_minj:c_maxj])[0]

    # test to see if this changes topology
    _new_ins_crop = new_ins_crop.copy()
    _new_ins_crop[p_mini:p_maxi, p_minj:p_maxj] = copy_ins

    before_ids = np.unique(new_ins_crop)
    after_ids = np.unique(_new_ins_crop)

    if (len(before_ids) != len(after_ids)) or not (before_ids == after_ids).all():
      # fig, [ax1, ax2] = plt.subplots(ncols=2)

      # ax1.imshow(new_ins_crop, cmap=utils.random_cmap)
      # ax2.imshow(_new_ins_crop, cmap=utils.random_cmap)

      # plt.show()
      # plt.close()
      continue

    num_before = len(np.unique(measure.label(new_ins_crop, background=-1,
                                             connectivity=1)))
    num_after = len(np.unique(measure.label(_new_ins_crop, background=-1,
                                            connectivity=1)))
    if num_before != num_after:
      # fig, [ax1, ax2] = plt.subplots(ncols=2)

      # ax1.imshow(new_ins_crop, cmap=utils.random_cmap)
      # ax2.imshow(_new_ins_crop, cmap=utils.random_cmap)

      # plt.show()
      # plt.close()
      continue

    new_ins_crop[p_mini:p_maxi, p_minj:p_maxj] = copy_ins
    new_sem_crop[p_mini:p_maxi, p_minj:p_maxj] = copy_sem

  return new_ins_crop, new_sem_crop


def shrink_binary_mask(mask):
  skeleton_mask = skeletonize(mask)
  eroded_mask = binary_erosion(mask, np.ones([3,3]))

  if not eroded_mask.sum():
    shrunk_mask = skeleton_mask
  else:
    original_area = float(mask.sum())
    shrunk_area = float(eroded_mask.sum())

    if (shrunk_area / original_area) < 0.5:
      shrunk_mask = skeleton_mask
    else:
      shrunk_mask = eroded_mask

  # make sure we didn't shrink the mask completely away
  assert shrunk_mask.max()

  shrunked = ~(shrunk_mask == mask).all()

  return shrunk_mask, shrunked


def expand_binary_mask(mask):
  return binary_dilation(mask, np.ones([3,3]))


def get_network_inputs(example, hide_prob, shrink_prob):
  crop_graph = example['crop_graph']

  # reshape
  ins_crop = utils.resize(example['ins_crop'], [64,64])
  sem_crop = utils.resize(example['sem_crop'], [64,64])
  img_crop = utils.resize(example['img_crop'], [64,64],
                          interpolation=interp_mode.BILINEAR)

  # set the order of the instance right now
  ins_ids = crop_graph.get_nodes_in_bbox(example['s_bbox'])

  # figure out which door symbols we can hide
  hide_ids = []
  hide_masks = {}

  for ins_id in crop_graph.G[example['fix_id']]:
    if (crop_graph.ins_sems[ins_id] == 7) and (ins_id in ins_ids):
      hide_flip = np.random.choice(hide_prob)

      # hide symbol and also mark where it roughly is
      if (ins_id not in ins_crop) or hide_flip:
        hide_mask = np.zeros_like(example['ins_crop'], dtype=bool)
        center_j, center_i = crop_graph.G.nodes[ins_id]['centroid']
        center_j, center_i = int(center_j), int(center_i)
        s_mini, s_minj, s_maxi, s_maxj = example['s_bbox']
        center_i -= s_mini
        center_j -= s_minj
        center_i += np.random.choice([-2, -1, 0, 1, 2])
        center_j += np.random.choice([-2, -1, 0, 1, 2])

        mini = max(center_i - 8, 0)
        maxi = min(center_i + 9, hide_mask.shape[0])
        minj = max(center_j - 8, 0)
        maxj = min(center_j + 9, hide_mask.shape[1])

        hide_mask[mini:maxi, minj:maxj] = True
        hide_mask = utils.resize(hide_mask, [64,64])

        # try hiding and make sure we are able to hide it
        ins_mask = (ins_crop == ins_id)
        ins_mask[hide_mask] = False

        if not ins_mask.any():
          hide_ids.append(ins_id)
          hide_masks[ins_id] = hide_mask

  # some inputs
  given_masks = []
  full_masks = []
  ind_masks = []
  ignore_masks = []
  semantics = []

  for ins_id in ins_ids:
    # GT instance mask
    ins_mask = (ins_crop == ins_id)
    full_mask = (ins_mask.astype(np.float32) - 0.5) * 2
    full_masks.append(full_mask)

    given_mask = full_mask.copy()

    # mark roughly where the symbol is
    if ins_id in hide_ids:
      given_mask[hide_masks[ins_id]] = 0
      assert not (given_mask > 0).any()
      given_masks.append(given_mask)
      ind_masks.append(1)

    # otherwise we hide a border around the instance
    else:
      shrunk_mask, shrunked = shrink_binary_mask(ins_mask)

      # only hide along edges if instance is big enough
      shrink_flip = np.random.choice(shrink_prob)
      if shrunked and shrink_flip:
        expand_mask = expand_binary_mask(ins_mask)
        ignore_mask = (shrunk_mask != expand_mask)
        given_mask[ignore_mask] = 0

      for hide_mask in hide_masks.values():
        if ins_mask[hide_mask].sum():
          given_mask[hide_mask] = 0

      given_masks.append(given_mask)
      ind_masks.append(0)

    # instance semantics
    semantics.append(crop_graph.ins_sems[ins_id])

    # shrunk mask for L2 loss
    l2_ignore_mask = (given_mask < 0.5) & (given_mask > -0.5)
    ignore_masks.append(l2_ignore_mask)

  given_masks = np.stack(given_masks)
  ignore_masks = np.stack(ignore_masks)
  full_masks = np.stack(full_masks)
  ind_masks = np.stack(ind_masks)

  # fully-connected graph for Conv-MPN
  graph_edges = []

  for k in range(len(semantics)):
    for l in range(len(semantics)):
      if l > k:
        graph_edges.append([k, 1, l])

  if not len(graph_edges):
    raise Exception('Empty input graph?')

  # normalize image crop first to [0,1], then [-1,1]
  img_crop -= img_crop.min()
  img_crop /= img_crop.max()

  img_crop = (img_crop - 0.5) * 2
  assert (img_crop.min() == -1) and (img_crop.max() == 1)

  # turn everything into PyTorch tensors
  semantics = one_hot_embedding(semantics, num_classes=8)

  given_masks = torch.FloatTensor(given_masks)
  ind_masks = torch.FloatTensor(ind_masks)
  ignore_masks = torch.BoolTensor(ignore_masks)
  given_imgs = torch.FloatTensor(img_crop)
  full_masks = torch.FloatTensor(full_masks)
  semantics = torch.FloatTensor(semantics)
  graph_edges = torch.LongTensor(graph_edges)

  sample = {
    'given_masks': given_masks,
    'semantics': semantics,
    'graph_edges': graph_edges,
    'given_imgs': given_imgs,
    'full_masks': full_masks,
    'ignore_masks': ignore_masks,
    'ind_masks': ind_masks
  }

  return sample


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
  order = ['top', 'right', 'bottom', 'left',
           'top_symbol', 'right_symbol', 'bottom_symbol', 'left_symbol']
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
    'left': left_edge
  }
  return edges


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


def get_symbol_hist(sem_ring):
  hist = np.zeros(8, dtype=np.int64)

  for (_, sem, _) in sem_ring:
    hist[sem] = 1.

  return hist


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

  top_symbol_ids = []
  right_symbol_ids = []
  bottom_symbol_ids = []
  left_symbol_ids = []

  for neighbor_segment in neighbor_segments:
    top_coverage = 0.
    right_coverage = 0.
    bottom_coverage = 0.
    left_coverage = 0.

    for neighbor_edge in neighbor_segment:
      # ignore edges that are on the perimeter
      if neighbor_edge[1] < 0:
        return {}

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

    # if we cover at least 35% of that edge, we also toggle it
    # also keep track of symbol topology
    added = False
    threshold = 0.25
    max_coverage = max(top_coverage, right_coverage,
                       bottom_coverage, left_coverage)

    if (max_coverage == top_coverage) or (top_coverage >= threshold):
      added = True
      top_hist[neighbor_semantic] = 1

      if (neighbor_semantic == 7) and (neighbor_id not in top_symbol_ids):
        _, symbol_sem_ring = all_sem_rings[neighbor_id]
        top_symbol_ids.append(neighbor_id)
        top_symbol_hist.append(get_symbol_hist(symbol_sem_ring))

    if (max_coverage == right_coverage) or (right_coverage >= threshold):
      added = True
      right_hist[neighbor_semantic] = 1

      if (neighbor_semantic == 7) and (neighbor_id not in right_symbol_ids):
        _, symbol_sem_ring = all_sem_rings[neighbor_id]
        right_symbol_ids.append(neighbor_id)
        right_symbol_hist.append(get_symbol_hist(symbol_sem_ring))

    if (max_coverage == bottom_coverage) or (bottom_coverage >= threshold):
      added = True
      bottom_hist[neighbor_semantic] = 1

      if (neighbor_semantic == 7) and (neighbor_id not in bottom_symbol_ids):
        _, symbol_sem_ring = all_sem_rings[neighbor_id]
        bottom_symbol_ids.append(neighbor_id)
        bottom_symbol_hist.append(get_symbol_hist(symbol_sem_ring))

    if (max_coverage == left_coverage) or (left_coverage >= threshold):
      added = True
      left_hist[neighbor_semantic] = 1

      if (neighbor_semantic == 7) and (neighbor_id not in left_symbol_ids):
        _, symbol_sem_ring = all_sem_rings[neighbor_id]
        left_symbol_ids.append(neighbor_id)
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
  tmp_edge = top_edge
  top_edge = bottom_edge
  bottom_edge = tmp_edge

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

    # plot the perimeter points
    # xx = [pt[1]-minj for pt in noncollinear_pts]
    # yy = [pt[0]-mini for pt in noncollinear_pts]
    # plt.plot(xx, yy, '-o', linewidth=1, markersize=3)

    # plot the four corners
    # plt.plot(noncollinear_pts[tl_idx][1]-minj, noncollinear_pts[tl_idx][0]-mini,
    #          '*', markersize=8)
    # plt.plot(noncollinear_pts[bl_idx][1]-minj, noncollinear_pts[bl_idx][0]-mini,
    #          '*', markersize=8)
    # plt.plot(noncollinear_pts[br_idx][1]-minj, noncollinear_pts[br_idx][0]-mini,
    #          '*', markersize=8)
    # plt.plot(noncollinear_pts[tr_idx][1]-minj, noncollinear_pts[tr_idx][0]-mini,
    #          '*', markersize=8)

    # plot the four-way histograms
    plt.title(topology_title(hist_dict))

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

  return hist_dict
