import numpy as np
import networkx as nx


NONE = 0
EDGE = 1
JUNC = 2
CORNER = 3


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


def get_graph(segmentation, pad=False):
  dtype = segmentation.dtype
  shape = segmentation.shape
  assert dtype == np.int64

  junction_map = np.zeros(shape, dtype=dtype)

  # when we're looking at small crops, in order to capture the junctions at
  # the border, we need a small padding
  if pad:
    segmentation = np.pad(segmentation, [[1,1], [1,1]], constant_values=-1)
    junction_map = np.pad(junction_map, [[1,1], [1,1]], constant_values=-1)

  # we should look at some border pixels as well
  # edge_map = sobel(segmentation.astype(np.float32))
  # edge_map = gaussian_filter(edge_map, sigma=0.5)

  if pad:
    edge_map[:, 0] = 1
    edge_map[0, :] = 1
    edge_map[:, -1] = 0
    edge_map[-1, :] = 0

  # edge_ii, edge_jj = np.nonzero(edge_map)
  for i in range(shape[0]-1):
    for j in range(shape[1]-1):
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

  G = nx.Graph()
  G.add_nodes_from(nodes)
  G.add_edges_from(edges)
  G = nx.relabel.convert_node_labels_to_integers(G, label_attribute='ij')

  return G
