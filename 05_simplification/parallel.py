import glob
import time
import multiprocessing as mp

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from contextlib import closing

from skimage.filters import sobel
from skimage.morphology import dilation, square
from scipy.ndimage import gaussian_filter

import utils

from timer import Timer


NONE = 0
EDGE = 1
JUNC = 2
CORNER = 3


def _init(shared_seg_, shared_jun_, dtype_, shape_):
  global shared_seg
  global shared_jun
  global dtype
  global shape

  shared_seg = shared_seg_
  shared_jun = shared_jun_
  dtype = dtype_
  shape = shape_


def shared_to_numpy(shared_arr, dtype, shape):
  return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


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


# TODO there are some checks that are kind of special, double-check them
def find_junctions(params):
  i, j = params
  seg_view = shared_to_numpy(shared_seg, dtype, shape)
  jun_view = shared_to_numpy(shared_jun, dtype, shape)

  block = seg_view[i:i+2, j:j+2]
  jun_view[i,j] = check_block(block)


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


def find_neighbors_wrapper(params):
  i, j = params

  seg_view = shared_to_numpy(shared_seg, dtype, shape)
  jun_view = shared_to_numpy(shared_jun, dtype, shape)

  return find_neighbors(i, j, seg_view, jun_view)


def test_check_block():
  block1 = np.array([[0, 0],
                     [0, 0]], dtype=np.float32)
  assert check_block(block1) == NONE

  block2 = np.array([[0, 0],
                     [1, 0]], dtype=np.float32)
  assert check_block(block2) == CORNER

  block3 = np.array([[0, 0],
                     [1, 1]], dtype=np.float32)
  assert check_block(block3) == EDGE

  block4 = np.array([[0, 2],
                     [1, 1]], dtype=np.float32)
  assert check_block(block4) == T_JUNC


def get_graph(segmentation, small_crop=False, multi=False):
  # assert small_crop == False
  assert multi == False

  dtype = segmentation.dtype
  shape = segmentation.shape
  assert dtype == np.int64

  junction_map = np.zeros(shape, dtype=dtype)


  """
    Small crop checking
  """
  if small_crop:
    # when we're looking at small crops, in order to capture the junctions at
    # the border, we need a small padding
    segmentation = np.pad(segmentation, [[1,1], [1,1]], constant_values=-1)
    junction_map = np.pad(junction_map, [[1,1], [1,1]], constant_values=-1)
    boundary_map = np.zeros_like(junction_map)

    # we should look at some border pixels as well
    edge_map = np.ones_like(segmentation, dtype=np.float32)
    edge_map[:, 0] = 1
    edge_map[0, :] = 1
    edge_map[:, -1] = 0
    edge_map[-1, :] = 0

  else:
    boundary_map = np.zeros_like(junction_map)
    edge_map = np.ones_like(segmentation, dtype=np.float32)
    edge_map[:, -1] = 0
    edge_map[-1, :] = 0
    # edge_map = sobel(segmentation.astype(np.float32))
    # edge_map = gaussian_filter(edge_map, sigma=0.5)


  """
    Single- or multi-threaded graph finding
  """
  if multi:
    cdtype = np.ctypeslib.as_ctypes_type(dtype)

    # make segmentation a shared array
    shared_seg = mp.RawArray(cdtype, shape[0] * shape[1])
    seg_view = shared_to_numpy(shared_seg, dtype, shape)
    np.copyto(seg_view, segmentation)

    # create a new shared array to hold junction types
    shared_jun = mp.RawArray(cdtype, shape[0] * shape[1])
    jun_view = shared_to_numpy(shared_jun, dtype, shape)
    np.copyto(jun_view, junction_map)

    # run all the multiprocessing steps
    with closing(mp.Pool(hparams['num_threads'], initializer=_init,
                         initargs=(shared_seg, shared_jun, dtype, shape))) as p:
      # populate junction matrix with junction types, but only check along edges
      edge_ii, edge_jj = np.nonzero(edge_map)
      jobs = list(zip(edge_ii, edge_jj))
      p.map(find_junctions, jobs)

      # find neighbors of all corners and multi-way junctions
      node_ii, node_jj = np.nonzero(jun_view > EDGE)
      node_locs = list(zip(node_ii.tolist(), node_jj.tolist()))
      adj_list = p.map(find_neighbors_wrapper, node_locs)

    p.join()

  else:
    edge_ii, edge_jj = np.nonzero(edge_map)
    for i,j in zip(edge_ii, edge_jj):
      block = segmentation[i:i+2, j:j+2]
      block_type = check_block(block)
      junction_map[i,j] = block_type

      if block_type > NONE:
        boundary_map[i:i+2, j:j+2] = 1

    adj_list = []
    node_ii, node_jj = np.nonzero(junction_map > EDGE)
    for i,j in zip(node_ii, node_jj):
      adj_list.append(find_neighbors(i, j, segmentation, junction_map))

    # unpad the boundary map
    # boundary_map = boundary_map[1:-1, 1:-1]

  # make a networkx graph
  nodes = []
  edges = []
  num_short_edges = 0

  for edge_list in adj_list:
    start = edge_list[0]
    nodes.append(start)

    for end in edge_list[1:]:
      edges.append((start, end))

      # edge_length = max(abs(start[0] - end[0]), abs(start[1] - end[1]))
      # if edge_length < hparams['short_edge_threshold']:
      #   num_short_edges += 1

  # # this is a special case when we are computing topology loss, in which we
  # # don't need a networkx graph or other things for that matter
  # if (not small_crop) and (not multi):
  #   return edges

  G = nx.Graph()
  G.add_nodes_from(nodes)
  G.add_edges_from(edges)
  G = nx.relabel.convert_node_labels_to_integers(G, label_attribute='ij')

  if small_crop:
    return nodes, edges
  else:
    return G, num_short_edges


if __name__ == '__main__':
  img_paths = glob.glob('./data/image/*.jpg')
  fp_ids = [x.strip().split('/')[-1][:-4] for x in img_paths]

  for fp_id in fp_ids:
    # threshold = 0.9 # 0.87 # 0.95 / # 10
    segmentation = utils.get_segmentation(fp_id)

    start = time.time()
    graph = get_graph(segmentation, num_threads=1)
    end = time.time()

    print('%s %f # nodes: %d # edges: %d' % (fp_id,
                                             end-start,
                                             len(graph.nodes()),
                                             len(graph.edges())))

    with open('times.csv', 'a') as f:
      f.write('%s %f # nodes: %d # edges: %d\n' % (fp_id,
                                                   end-start,
                                                   len(graph.nodes()),
                                                   len(graph.edges())))
