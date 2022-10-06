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
from skimage.filters import sobel
from shapely.geometry import Polygon
from scipy.ndimage import gaussian_filter

import paths


NONE = 0
EDGE = 1
JUNC = 2
CORNER = 3


# random cmap for visualization of instances
random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))


def check_block(block):
  # the entire square is one color
  if block[0,0] == block[0,1] == block[1,0] == block[1,1]:
    return NONE

  # two cases for edge
  elif (block[0,0] == block[0,1] and block[1,0] == block[1,1]) \
      or (block[0,0] == block[1,0] and block[0,1] == block[1,1]):
    return EDGE

  # four cases for corner
  elif (block[0,0] == block[0,1] == block[1,0]) \
      or (block[0,0] == block[0,1] == block[1,1]) \
      or (block[0,0] == block[1,0] == block[1,1]) \
      or (block[1,0] == block[1,1] == block[0,1]):
    return CORNER

  # rest are junctions
  else:
    return JUNC


def find_neighbors(i, j, segmentation, junction_map):
  # list used to keep track of found neighbors
  neighbors = [(i,j),]

  # look up
  if segmentation[i,j] != segmentation[i,j+1]:
    up_i = i - 1
    while up_i >= 0:
      neighbor_type = junction_map[up_i, j]

      if neighbor_type in [JUNC, CORNER]:
        neighbors.append((up_i, j))
        break

      elif neighbor_type == NONE:
        break

      else:
        up_i -= 1

  # look down
  if segmentation[i+1,j] != segmentation[i+1,j+1]:
    down_i = i + 1
    while down_i < junction_map.shape[0]:
      neighbor_type = junction_map[down_i, j]

      if neighbor_type in [JUNC, CORNER]:
        neighbors.append((down_i, j))
        break

      elif neighbor_type == NONE:
        break

      else:
        down_i += 1

  # look left
  if segmentation[i,j] != segmentation[i+1,j]:
    left_j = j - 1
    while left_j >= 0:
      neighbor_type = junction_map[i, left_j]

      if neighbor_type in [JUNC, CORNER]:
        neighbors.append((i, left_j))
        break

      elif neighbor_type == NONE:
        break

      else:
        left_j -= 1

  # look right
  if segmentation[i,j+1] != segmentation[i+1,j+1]:
    right_j = j + 1
    while right_j < junction_map.shape[1]:
      neighbor_type = junction_map[i, right_j]

      if neighbor_type in [JUNC, CORNER]:
        neighbors.append((i, right_j))
        break

      elif neighbor_type == NONE:
        break

      else:
        right_j += 1

  return neighbors


def get_graph(segmentation, pad=True):
  dtype = segmentation.dtype
  shape = segmentation.shape
  assert dtype == np.int64

  junction_map = np.zeros(shape, dtype=dtype)

  # when we're looking at small crops, in order to capture the junctions at
  # the border, we need a small padding
  if pad:
    segmentation = np.pad(segmentation, [[1,1], [1,1]], constant_values=0)
    junction_map = np.pad(junction_map, [[1,1], [1,1]], constant_values=0)

  # we should look at some border pixels as well
  # NOTE using gaussian blur because some pixels are not detected for reasons
  edge_map = sobel(segmentation.astype(np.float32))
  edge_map = gaussian_filter(edge_map, sigma=0.5)

  if pad:
    edge_map[:, 0] = 1
    edge_map[0, :] = 1
    edge_map[:, -1] = 0
    edge_map[-1, :] = 0

  edge_ii, edge_jj = np.nonzero(edge_map)
  for i,j in zip(edge_ii, edge_jj):
    block = segmentation[i:i+2, j:j+2]
    junction_map[i,j] = check_block(block)

  adj_list = []
  node_ii, node_jj = np.nonzero(junction_map > EDGE)
  for i,j in zip(node_ii, node_jj):
    adj_list.append(find_neighbors(i, j, segmentation, junction_map))

  # make a graph
  nodes = []
  edges = []

  for edge_list in adj_list:
    start = edge_list[0]

    if not pad:
      start = (start[0]+1, start[1]+1)

    nodes.append(start)

    for end in edge_list[1:]:
      if not pad:
        end = (end[0]+1, end[1]+1)

      if (end, start) not in edges:
        edges.append((start, end))

  return nodes, edges


def plot_edges(all_instance_mask, instance_edges):
  plt.imshow(all_instance_mask, cmap=random_cmap)

  for (a,b) in instance_edges:
    plt.plot([a[1]-0.5, b[1]-0.5], [a[0]-0.5, b[0]-0.5], '-o')

  plt.axis('off')
  plt.tight_layout()
  plt.show()


def get_instance_semantic(instance_id, all_instance_mask, all_semantic_mask):
  instance_mask = (all_instance_mask == instance_id)
  semantic_mask = all_semantic_mask[instance_mask]
  unique, counts = np.unique(semantic_mask, return_counts=True)
  instance_sem = unique[np.argmax(counts)]

  return instance_sem


def is_ccw(ring):
  points = [a for ((a,b), _, _) in ring]
  points.append(points[0])

  # NOTE the points have to loop back to itself for this to work
  assert points[0] == points[-1]

  area = 0

  for p1, p2 in zip(points[:-1], points[1:]):
    area += (p2[1] - p1[1]) * (p2[0] + p1[0])

  if area > 0:
    return False
  elif area < 0:
    return True
  else:
    raise Exception('Zero area polygon?')


def get_instance_edge_mapping(all_instance_mask, all_semantic_mask, pad=True):
  nodes, edges = get_graph(all_instance_mask, pad=pad)
  h, w = all_instance_mask.shape
  instance_to_edges = {}

  for (a,b) in edges:
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

    neighbor_ids = np.unique(all_instance_mask[n_mini:n_maxi, n_minj:n_maxj])

    for neighbor_id in neighbor_ids:
      if neighbor_id not in instance_to_edges.keys():
        instance_to_edges[neighbor_id] = []

      instance_to_edges[neighbor_id].append((a,b))

  ordered_instance_to_edges = {}

  # connect the list of edges and make sure they go ccw
  for instance_id in instance_to_edges.keys():
    # only windows, doors, frames, and door-handles for now
    if get_instance_semantic(instance_id,
                             all_instance_mask,
                             all_semantic_mask) not in [3,4,5,7]:
      ordered_instance_to_edges[instance_id] = []
      continue

    old_instance_edges = instance_to_edges[instance_id].copy()
    new_instance_edges = []

    # find the edge on the outer boundary
    min_idx = -1
    min_dist = float('inf')

    for idx, edge in enumerate(old_instance_edges):
      (a_x, a_y), (b_x, b_y) = edge
      dist = min(np.sqrt(a_x ** 2 + a_y ** 2), np.sqrt(b_x ** 2 + b_y ** 2))

      if dist < min_dist:
        min_idx = idx
        min_dist = dist

    curr_edge = old_instance_edges[min_idx]
    old_instance_edges.remove(curr_edge)
    new_instance_edges.append(curr_edge)

    while len(old_instance_edges):
      # look for the next edge that starts with b
      next_edge = None

      for candidate_edge in old_instance_edges:
        if curr_edge[1] in candidate_edge:
          if next_edge:
            next_edge = None
            break
          else:
            next_edge = candidate_edge

      # this means there's a hole in this instance, so skip
      if not next_edge:
        # plot_edges(all_instance_mask, new_instance_edges)
        break

      if curr_edge[1] == next_edge[0]:
        new_instance_edges.append((next_edge[0], next_edge[1]))
      elif curr_edge[1] == next_edge[1]:
        new_instance_edges.append((next_edge[1], next_edge[0]))
      else:
        raise Exception

      old_instance_edges.remove(next_edge)
      curr_edge = new_instance_edges[-1]

    # same as above, this means there's a hole in this instance
    if not len(new_instance_edges):
      raise Exception

    # orient the edges so that they are ccw
    # if not is_ccw(new_instance_edges):
    #   new_instance_edges = new_instance_edges[::-1]
    #   new_instance_edges = [(b,a) for (a,b) in new_instance_edges]
    #   assert is_ccw(new_instance_edges)

    # make sure the edges form a loop
    if new_instance_edges[-1][1] != new_instance_edges[0][0]:
      raise Exception('broken loop')

    # make sure the polygon is valid
    poly_coords = [(a[1], a[0]) for (a,b) in new_instance_edges]
    assert Polygon(poly_coords).is_valid

    # DEBUG visualize the ordered edges
    if False:
      plt.imshow(all_instance_mask, cmap=random_cmap)

      for (a,b) in new_instance_edges:
        plt.arrow(a[1], a[0], b[1]-a[1], b[0]-a[0], width=0.1)

      plt.axis('off')
      plt.tight_layout()
      plt.show()

    # save this ordered list of edges
    ordered_instance_to_edges[instance_id] = new_instance_edges

  return ordered_instance_to_edges


def get_sem_rings(instance_to_edges, all_instance_mask, all_semantic_mask):
  all_sem_rings = {}
  h, w = all_instance_mask.shape

  for instance_id, instance_edges in instance_to_edges.items():
    instance_ring = []

    for (a,b) in instance_edges:
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

      neighbor_ids = np.unique(all_instance_mask[n_mini:n_maxi, n_minj:n_maxj])

      if len(neighbor_ids) == 1:
        # this edge is on the perimeter, so no neighbor information
        assert instance_id in neighbor_ids
        instance_ring.append((((ai,aj), (bi,bj)), -1, -1))

      else:
        assert (len(neighbor_ids) == 2) and (instance_id in neighbor_ids)

        # we always want to grab the semantic on the other side of this instance
        if instance_id == neighbor_ids[0]:
          neighbor_id = neighbor_ids[1]
        else:
          neighbor_id = neighbor_ids[0]

        neighbor_sem = get_instance_semantic(neighbor_id,
                                             all_instance_mask,
                                             all_semantic_mask)
        instance_ring.append((((ai,aj), (bi,bj)),
                              int(neighbor_sem), int(neighbor_id)))

    instance_sem = get_instance_semantic(instance_id,
                                         all_instance_mask,
                                         all_semantic_mask)

    assert int(instance_id) not in all_sem_rings.keys()
    all_sem_rings[int(instance_id)] = [int(instance_sem), instance_ring]

  return all_sem_rings


def get_edge_orientation(edge):
  (ai,aj), (bi,bj) = edge

  if ai == bi:
    orientation = 'h'
  elif aj == bj:
    orientation = 'v'
  else:
    raise Exception

  return orientation


def get_edge_length(edge):
  (ai,aj), (bi,bj) = edge

  # make sure this edge is manhattan and of some length
  assert ((ai-bi) + (aj-bj) != 0) and ((ai-bi) * (aj-bj) == 0)

  return max(abs(ai-bi), abs(aj-bj))


def get_long_edges(old_groups, threshold=2):
  new_edges = []

  # now group edges together while ignoring short edges
  for old_group in old_groups:
    prev_orientation = None
    edges = []

    for edge in old_group:
      edge_len = get_edge_length(edge[0])
      curr_orientation = get_edge_orientation(edge[0])

      if prev_orientation == None:
        prev_orientation = curr_orientation
        edges.append(edge)

      else:
        if edge_len < threshold:
          edges.append(edge)
        else:
          if curr_orientation == prev_orientation:
            edges.append(edge)
          else:
            new_edges.append(edges)
            prev_orientation = curr_orientation
            edges = [edge,]

    new_edges.append(edges)

  return new_edges


def get_neighbor_edges(instance_ring):
  # rotate instance ring so we start at a new edge
  pivot = 0
  prev_id = instance_ring[0][2]
  for idx, (_, _, next_id) in enumerate(instance_ring):
    if next_id != prev_id:
      pivot = idx
      break
  instance_ring = instance_ring[pivot:] + instance_ring[:pivot]

  # now do some stuff
  new_instance_ring = []
  prev_id = None
  edges = []

  for edge, neighbor_sem, neighbor_id in instance_ring:
    if prev_id == None:
      prev_id = neighbor_id
      edges.append((edge, neighbor_sem, neighbor_id))

    elif prev_id == neighbor_id:
      edges.append((edge, neighbor_sem, neighbor_id))

    elif prev_id != neighbor_id:
      new_instance_ring.append(edges)
      prev_id = neighbor_id
      edges = [(edge, neighbor_sem, neighbor_id)]

    else:
      raise Exception

  new_instance_ring.append(edges)

  return new_instance_ring


def get_gt_sem_rings(fp_id):
  print(fp_id)

  instance_f = paths.INSTANCE_ROOT + '%s.npy' % fp_id
  semantic_f = paths.SEMANTIC_ROOT + '%s.npy' % fp_id

  all_instance_mask = np.load(instance_f, allow_pickle=False)
  all_semantic_mask = np.load(semantic_f, allow_pickle=False)

  # woops semantic class is kind of inconsistent...
  all_semantic_mask[all_semantic_mask == 24] = 7

  # alright, start doing some work here
  instance_to_edges = get_instance_edge_mapping(all_instance_mask,
                                                all_semantic_mask)

  all_sem_rings = get_sem_rings(instance_to_edges,
                                all_instance_mask,
                                all_semantic_mask)

  # save it to a JSON file
  with open(paths.PREPROCESS_ROOT + 'sem_rings/%s_gt.json' % fp_id, 'w') as f:
    json.dump(all_sem_rings, f)


def get_pred_sem_rings(fp_id):
  print(fp_id)

  instance_f = paths.PRED_INSTANCE_ROOT + '%s.npy' % fp_id
  semantic_f = paths.PRED_SEMANTIC_ROOT + '%s.npy' % fp_id

  all_instance_mask = np.load(instance_f, allow_pickle=False)
  all_semantic_mask = np.load(semantic_f, allow_pickle=False)

  # woops semantic class is kind of inconsistent...
  all_semantic_mask[all_semantic_mask == 24] = 7

  # alright, start doing some work here
  instance_to_edges = get_instance_edge_mapping(all_instance_mask,
                                                all_semantic_mask)

  all_sem_rings = get_sem_rings(instance_to_edges,
                                all_instance_mask,
                                all_semantic_mask)

  # save it to a JSON file
  with open(paths.PREPROCESS_ROOT + 'sem_rings/%s_pred.json' % fp_id, 'w') as f:
    json.dump(all_sem_rings, f)


def resave_pred_instance_masks(fp_ids):
  for fp_id in fp_ids:
    semantic_f = paths.PRED_SEMANTIC_ROOT + '%s.npy' % fp_id
    instance_f = paths.PRED_INSTANCE_ROOT + '%s.npy' % fp_id

    all_semantic_mask = np.load(semantic_f, allow_pickle=False)
    all_instance_mask = measure.label(all_semantic_mask,
                                      background=-1,
                                      connectivity=1)

    np.save(instance_f, all_instance_mask)


if __name__ == '__main__':
  from multiprocessing import Pool

  if os.path.exists(paths.PREPROCESS_ROOT + 'sem_rings/'):
    shutil.rmtree(paths.PREPROCESS_ROOT + 'sem_rings/')
  os.mkdir(paths.PREPROCESS_ROOT + 'sem_rings/')

  semantic_files = glob.glob(paths.PRED_SEMANTIC_ROOT + '*.npy')
  fp_ids = [x.strip().split('/')[-1].split('.')[0] for x in semantic_files]

  # resave_pred_instance_masks(fp_ids)

  # multi-threaded
  with Pool(10) as p:
    p.map(get_gt_sem_rings, fp_ids)

  with Pool(10) as p:
    p.map(get_pred_sem_rings, fp_ids)
