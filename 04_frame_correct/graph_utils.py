import matplotlib
# matplotlib.use('Agg')

import os
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from skimage import measure
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union, nearest_points
from shapely.algorithms.polylabel import polylabel

import utils
import sem_rings
import data_utils


class CropGraph:

  def __init__(self, ins_crop, sem_crop, fp_id,
               fix_id, remove_ids, topo_vec, fix):
    # save everything
    self.fp_id = fp_id
    self.fix_id = fix_id
    self.remove_ids = remove_ids
    self.ins_crop = ins_crop
    self.sem_crop = sem_crop
    self.topo_vec = topo_vec

    self.fix = fix
    self.symbol_cases = {}
    self.warnings = []

    self.side_fixed = {}
    self.fix_focus = {}

    # collect any erroneous instance as well
    self.all_ins_edges = sem_rings.get_instance_edge_mapping(self.ins_crop,
                                                             self.sem_crop)
    self.all_sem_rings = sem_rings.get_sem_rings(self.all_ins_edges,
                                                 self.ins_crop,
                                                 self.sem_crop)

    for ins_id, ins_edges in self.all_ins_edges.items():
      if not len(ins_edges):
        self.warnings.append('graph, get ins edges, %d bad' % ins_id)

    # construct instance semantic mapping
    self.ins_sems = {}
    for ins_id, (ins_sem, _) in self.all_sem_rings.items():
      self.ins_sems[ins_id] = ins_sem

    # get the four sides to the center instance
    if self.fix_id > 0:
      self.sides = self.get_four_sides(self.fix_id)

		# construct graph from the crop raw
    nodes, edges = sem_rings.get_graph(ins_crop)
    h,w = ins_crop.shape
    self.G = nx.Graph()

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

      neighbors = np.unique(ins_crop[n_mini:n_maxi, n_minj:n_maxj])

      # normal boundary
      if len(neighbors) == 2:
        (a,b) = neighbors.tolist()

        self.G.add_edge(a, b, new=False)
        self.G.nodes[a]['new'] = False
        self.G.nodes[b]['new'] = False

      # boundary along edge of image
      elif len(neighbors) == 1:
        a = neighbors.tolist()[0]
        if not self.G.has_node(a):
          self.G.add_node(a)
          self.G.nodes[a]['new'] = False

      else:
        raise Exception('cannot have more than 2 neighbors!')

    # mark the centroid of each node
    centroids = {}

    for ins_id in np.unique(self.ins_crop):
      if len(self.all_ins_edges[ins_id]):
        poly_coords = [(a[1], a[0]) for (a,b) in self.all_ins_edges[ins_id]]
        polygon = Polygon(poly_coords)
        centroid = polylabel(polygon)
        centroids[ins_id] = {'centroid': [centroid.x, centroid.y]}
      else:
        centroids[ins_id] = {'centroid': [0, 0]}

    nx.set_node_attributes(self.G, centroids)

    # get the neighbor IDs around the target instance to fix
    neighbor_ids = {
      'top': set(),
      'right': set(),
      'bottom': set(),
      'left': set(),
    }

    for key in ['top', 'right', 'bottom', 'left']:
      self.fix_focus[key] = []

      for _, neighbor_sem, neighbor_id in self.sides[key]:
        neighbor_ids[key].add(neighbor_id)

        # point of interest for later on
        if neighbor_sem == 7:
          self.fix_focus[key].append(self.G.nodes[neighbor_id]['centroid'])

    # if we don't need to fix anything, then we're done
    if not fix:
      return

    # update the graph around the target instance to fix
    before_vec = np.array(self.topo_vec['before'], dtype=int)
    after_vec = np.round(self.topo_vec['after']).astype(int)

    before_hist = data_utils.restore_dict(before_vec)
    after_hist = data_utils.restore_dict(after_vec)
    self.before_hist = before_hist
    self.after_hist = after_hist

    symbol_ids_by_sides = self.get_symbol_neighbors(self.fix_id)

    for key in ['top', 'right', 'bottom', 'left']:
      """
        check symbols

        Alright alright, what's below is probably the most confusing part of
        this class. I'm iterating through all the combination of adding or
        removing or fixing symbols, so whoever is reading this code, yes, you,
        I wish you good luck!
      """
      num_symbols_before = before_hist[key].sum()
      num_symbols_after = after_hist[key].sum()
      assert num_symbols_before == len(symbol_ids_by_sides[key])

      # NOTE more of a debugging step, determine which case it is for symbols
      symbol_case = '%d_%d' % (num_symbols_before, num_symbols_after)
      self.symbol_cases[key] = symbol_case

      self.side_fixed[key] = False

      # we need to add
      if not num_symbols_before and num_symbols_after:
        if num_symbols_after == 2:
          import pdb; pdb.set_trace()

        new_id = max(self.G.nodes) + 1
        self.side_fixed[key] = True

        # add all the new edges
        self.G.add_edge(self.fix_id, new_id, new=True)
        self.G.nodes[new_id]['new'] = True
        self.ins_sems[new_id] = 7

        # find roughly where we need to focus
        side_coords = [x[0][0][::-1] for x in self.sides[key]]
        side_coords.append(self.sides[key][-1][0][1][::-1])
        side_center = LineString(side_coords).interpolate(0.5, normalized=True)

        wall_polys = []
        for neighbor_id in self.get_neighbors_along_side(self.sides[key]):
          neighbor_sem = self.ins_sems[neighbor_id]
          if neighbor_sem in [1,2]:
            wall_coords = [(a[1], a[0]) for (a,b) in self.all_ins_edges[neighbor_id]]
            wall_polys.append(Polygon(wall_coords))

        if not len(wall_polys):
          rough_centroid = np.array(side_center)
        else:
          wall_poly = unary_union(wall_polys)
          rough_centroid = np.array(nearest_points(wall_poly, side_center)[0])

        self.fix_focus[key].append(rough_centroid)
        self.G.nodes[new_id]['centroid'] = rough_centroid.tolist()

      # we need to remove
      elif num_symbols_before and not num_symbols_after:
        # operate on only symbols
        for symbol_id in symbol_ids_by_sides[key]:
          # need to look around this symbol
          self.fix_focus[key].append(self.G.nodes[symbol_id]['centroid'])

          # this symbol is not needed by any other instance
          if symbol_id in remove_ids:
            self.G.remove_node(symbol_id)
            self.side_fixed[key] = True

      # we need to fix and maybe add or remove
      elif num_symbols_before and num_symbols_after:
        # easy case
        if (num_symbols_before == 1) and (num_symbols_after == 1):
          pass

        elif (num_symbols_before == 2) and (num_symbols_after == 2):
          pass

        elif (num_symbols_before == 2) and (num_symbols_after == 1):
          assert len(symbol_ids_by_sides[key]) == 2
          self.side_fixed[key] = True

          # figure out which symbol we need to remove, if any
          symbol_id_0, symbol_id_1 = symbol_ids_by_sides[key]

          if symbol_id_0 in remove_ids:
            assert symbol_id_1 not in remove_ids
            self.G.remove_node(symbol_id_0)

          elif symbol_id_1 in remove_ids:
            assert symbol_id_0 not in remove_ids
            self.G.remove_node(symbol_id_1)

          else:
            pass

        elif (num_symbols_before == 1) and (num_symbols_after == 2):
          assert len(symbol_ids_by_sides[key]) == 1
          self.side_fixed[key] = True
          # NOTE need to make the binary symbol the same

          # this is a special case, the missing door symbol needs to touch
          # the other inner wall that the existing symbol is not touching
          wall_candidates = []
          for neighbor_id in neighbor_ids[key]:
            if self.ins_sems[neighbor_id] == 2:
              wall_candidates.append(neighbor_id)

          existing_symbol_id = symbol_ids_by_sides[key][0]
          for neighbor_id in self.G[existing_symbol_id]:
            if neighbor_id in wall_candidates:
              wall_candidates.remove(neighbor_id)

          # if we don't have two wall candidates, then we're missing a wall,
          # that's too complicated, so skip...
          if len(wall_candidates) != 1:
            self.warnings.append('graph, fix 1_2, only one wall candidate')
            continue
          wall_id = wall_candidates[0]

          new_id = max(self.G.nodes) + 1

          # add all the new edges
          self.G.add_edge(self.fix_id, new_id, new=True)
          self.G.nodes[new_id]['new'] = True
          self.ins_sems[new_id] = 7

          # get the rough centroid of this new instance
          side_coords = [x[0][0][::-1] for x in self.sides[key]]
          side_coords.append(self.sides[key][-1][0][1][::-1])
          side_center = LineString(side_coords).interpolate(0.5, normalized=True)
          self.G.nodes[new_id]['centroid'] = np.array(side_center).tolist()

        else:
          pass  # NOTE this is weird
          # raise Exception('Unknown symbol fixing combination')


  def get_neighbors_along_side(self, side, margin=2):
    neighbors = set()
    h,w = self.ins_crop.shape

    for ((a,b), n_sem, n_id) in side:
      ai, aj = int(a[0]), int(a[1])
      bi, bj = int(b[0]), int(b[1])

      # this is a horizontal edge
      if ai == bi:
        n_mini = max(ai - margin, 0)
        n_maxi = min(ai + margin, h)
        n_minj = max(min(aj, bj) - margin, 0)
        n_maxj = min(max(aj, bj) + margin, w)

      # this is a vertical edge
      elif aj == bj:
        n_mini = max(min(ai, bi) - margin, 0)
        n_maxi = min(max(ai, bi) + margin, h)
        n_minj = max(aj - margin, 0)
        n_maxj = min(aj + margin, w)

      else:
        raise Exception('non-manhattan edge')

      for neighbor in np.unique(self.ins_crop[n_mini:n_maxi, n_minj:n_maxj]):
        neighbors.add(int(neighbor))

      # DEBUG to see which area we are looking at
      if False:
        highlight = np.zeros_like(self.ins_crop)
        highlight[n_mini:n_maxi, n_minj:n_maxj] = 1.

        plt.imshow(self.sem_crop / 7., cmap='nipy_spectral',
                   vmin=0., vmax=1., interpolation='nearest')
        plt.imshow(highlight, cmap='gray', alpha=0.7)
        plt.show()
        plt.close()

    return neighbors


  def get_symbol_neighbors(self, ins_id):
    semantic, instance_ring = self.all_sem_rings[ins_id]
    assert semantic in [3,4,5]

    edges = data_utils.get_four_sides_2(instance_ring)
    top_edge = edges['top']
    right_edge = edges['right']
    bottom_edge = edges['bottom']
    left_edge = edges['left']

    # split the semantic ring by instance
    prev_id = -1
    prev_segment = []
    neighbor_segments = []

    for (n_edge, n_sem, n_id) in data_utils.rotate_to_new_instance(instance_ring):
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

    # my goodness...
    top_ids = []
    right_ids = []
    bottom_ids = []
    left_ids = []

    for neighbor_segment in neighbor_segments:
      neighbor_semantic = [x[1] for x in neighbor_segment]
      assert len(np.unique(neighbor_semantic)) == 1
      neighbor_semantic = neighbor_semantic[0]

      if neighbor_semantic != 7:
        continue

      neighbor_id = [x[2] for x in neighbor_segment]
      assert len(np.unique(neighbor_id)) == 1
      neighbor_id = neighbor_id[0]

      top_coverage = 0.
      right_coverage = 0.
      bottom_coverage = 0.
      left_coverage = 0.

      for neighbor_edge in neighbor_segment:
        # ignore edges that are on the perimeter
        if neighbor_edge[1] < 0:
          raise Exception

        dist = data_utils.get_edge_length(neighbor_edge[0])

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
      top_coverage /= data_utils.get_segment_length(top_edge)
      right_coverage /= data_utils.get_segment_length(right_edge)
      bottom_coverage /= data_utils.get_segment_length(bottom_edge)
      left_coverage /= data_utils.get_segment_length(left_edge)

      # if we cover at least 25% of that edge, we also toggle it
      # also keep track of symbol topology
      added = False
      max_coverage = max(top_coverage, right_coverage,
                         bottom_coverage, left_coverage)

      if (max_coverage == top_coverage) or (top_coverage >= 0.25):
        top_ids.append(neighbor_id)

      if (max_coverage == right_coverage) or (right_coverage >= 0.25):
        right_ids.append(neighbor_id)

      if (max_coverage == bottom_coverage) or (bottom_coverage >= 0.25):
        bottom_ids.append(neighbor_id)

      if (max_coverage == left_coverage) or (left_coverage >= 0.25):
        left_ids.append(neighbor_id)

    # origin is top-left, so up and down are backwards
    results = {
      'top': list(set(bottom_ids)),
      'right': list(set(right_ids)),
      'bottom': list(set(top_ids)),
      'left': list(set(left_ids))
    }
    return results


  def get_neighbors_near(self, ins_id, margin=5):
    ins_mask = (self.ins_crop == ins_id)

    ii, jj = np.nonzero(ins_mask)
    height = ii.max() - ii.min() + 2 * margin
    width = jj.max() - jj.min() + 2 * margin
    center_i = (ii.max() + ii.min()) // 2
    center_j = (jj.max() + jj.min()) // 2

    mini = max(center_i - height // 2, 0)
    minj = max(center_j - width // 2, 0)
    mini = min(mini, self.ins_crop.shape[0] - height)
    minj = min(minj, self.ins_crop.shape[1] - width)

    maxi = min(mini + height, self.ins_crop.shape[0])
    maxj = min(minj + width, self.ins_crop.shape[1])

    assert (mini >= 0) and (minj >= 0)
    assert (maxi <= self.ins_crop.shape[0]) and (maxj <= self.ins_crop.shape[1])

    neighbors = np.unique(self.ins_crop[mini:maxi, minj:maxj])

    # DEBUG check the area we're getting neighbors from
    if False:
      binary_mask = np.zeros_like(self.sem_crop)
      binary_mask[mini:maxi, minj:maxj] = 1

      plt.imshow(self.sem_crop / 7., cmap='nipy_spectral',
                 vmin=0., vmax=1., interpolation='nearest')
      plt.imshow(binary_mask, cmap='gray', alpha=0.7)
      plt.show()
      plt.close()

    return neighbors


  def get_graph(self, ins_order):
    graph_edges = []
    mapping = dict(zip(ins_order, range(len(ins_order))))

    for ins_a in ins_order:
      for ins_b in ins_order:
        if ins_a > ins_b:
          if self.G.has_edge(ins_a, ins_b):
            graph_edges.append([mapping[ins_a], 1, mapping[ins_b]])
          else:
            graph_edges.append([mapping[ins_a], -1, mapping[ins_b]])

    return graph_edges


  def get_four_sides(self, target_id):
    semantic, instance_ring = self.all_sem_rings[target_id]

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
      'top': bottom_edge,
      'right': right_edge,
      'bottom': top_edge,
      'left': left_edge
    }
    return edges


  def get_crops(self):
    h,w = self.ins_crop.shape
    assert h == w
    crops = []

    # if the crop is small enough, just return it
    if h <= 200:
      crops.append((self.ins_crop, self.sem_crop, [0, 0, h, w]))
      return crops

    for key in ['top', 'right', 'bottom', 'left']:
      _fix_focus = [tuple(x) for x in self.fix_focus[key]]
      for center in set(_fix_focus):
        # compute bounding box around center
        side_len = 128
        center_j, center_i = np.round(center).astype(int)
        mini = max(center_i - side_len // 2, 0)
        minj = max(center_j - side_len // 2, 0)
        mini = min(mini, self.ins_crop.shape[0] - side_len)
        minj = min(minj, self.ins_crop.shape[1] - side_len)

        maxi = mini + side_len
        maxj = minj + side_len

        assert (mini >= 0) and (minj >= 0)
        assert (maxi <= self.ins_crop.shape[0]) and (maxj <= self.ins_crop.shape[1])

        # crop
        crops.append((self.ins_crop[mini:maxi, minj:maxj].copy(), \
                      self.sem_crop[mini:maxi, minj:maxj].copy(), \
                      [mini, minj, maxi, maxj]))

        if False:
          fig = plt.figure(figsize=(12,8))
          plt.imshow(self.sem_crop / 7., cmap='nipy_spectral',
                     vmin=0., vmax=1., interpolation='nearest')
          plt.imshow(self.ins_crop, alpha=0)

          # plot the side of interest
          xx, yy = [], []
          for edge, _, _ in self.sides[key]:
            xx.append(edge[0][1]-0.5)
            xx.append(edge[1][1]-0.5)
            yy.append(edge[0][0]-0.5)
            yy.append(edge[1][0]-0.5)

          plt.plot(xx, yy, '-o', linewidth=2, markersize=3, label=key)

          # plot the bbox around this region
          # coords = np.array(list(set(zip(xx, yy))))
          # center = coords.mean(axis=0)
          plt.plot(center[0], center[1], '*', markersize=10)

          plt.plot([minj, maxj, maxj, minj, minj],
                   [mini, mini, maxi, maxi, mini], '-c')

          # see if the side is within bounds or not
          xx, yy = np.array(xx), np.array(yy)
          out_of_bounds = np.any([xx < minj, xx > maxj, yy < mini, yy > maxi])

          if out_of_bounds:
            save_f = './debug/out/%s_%d.png' % (self.fp_id, self.fix_id)
          else:
            save_f = './debug/in/%s_%d.png' % (self.fp_id, self.fix_id)

          os.makedirs(os.path.dirname(save_f), exist_ok=True)
          plt.axis('off')
          plt.tight_layout()
          # plt.savefig(save_f)
          plt.show()
          plt.close()

          self.vis_graph()

    return crops


  def get_nodes_in_bbox(self, bbox, margin=4):
    return_nodes = []
    miny, minx, maxy, maxx = bbox

    # get a reshaped instance crop to avoid reshaped out nodes
    ins_crop = self.ins_crop[miny:maxy, minx:maxx].copy()
    ins_crop = utils.resize(ins_crop, [64,64])

    for ins_id in self.G.nodes:
      if ins_id in ins_crop:
        return_nodes.append(ins_id)

      else:
        if ins_id not in self.ins_crop:
          (x,y) = self.G.nodes[ins_id]['centroid']
          if (x >= minx) and (x <= maxx) and (y >= miny) and (y <= maxy):
            return_nodes.append(ins_id)

    return return_nodes


  def debug_vis(self):
    plt.figure(figsize=(12,8))
    plt.imshow(self.sem_crop / 7., cmap='nipy_spectral',
               vmin=0., vmax=1., interpolation='nearest')
    plt.imshow(self.ins_crop, alpha=0)

    plt.axis('off')
    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(0.001)


  def vis_graph(self, fname=None):
    fig = plt.figure(figsize=(12,8))
    plt.imshow(self.sem_crop / 7., cmap='nipy_spectral',
               vmin=0., vmax=1., interpolation='nearest')
    plt.imshow(self.ins_crop, alpha=0)

    # plot the four sides
    for key in ['top', 'right', 'bottom', 'left']:
      xx, yy = [], []
      for edge, _, _ in self.sides[key]:
        xx.append(edge[0][1]-0.5)
        xx.append(edge[1][1]-0.5)
        yy.append(edge[0][0]-0.5)
        yy.append(edge[1][0]-0.5)

      plt.plot(xx, yy, '-o', linewidth=2, markersize=3, label=key)

    if 0 in self.sem_crop:
      color = 'c'
    else:
      color = 'k'

    # plot each instance's centroid
    focus_ids = [self.fix_id,]

    for a in self.G.nodes:
      a_x, a_y = self.G.nodes[a]['centroid']

      if self.ins_sems[a] == 7:
        focus_ids.append(a)

      if a not in self.ins_crop:
        plt.plot(a_x, a_y, '*'+color, markersize=10)
      else:
        plt.plot(a_x, a_y, 'o'+color, markersize=5)

    if self.fix:
      title_str = '%s %d\n\n' % (self.fp_id, self.fix_id)
      for key in ['top', 'right', 'bottom', 'left']:
        sem_vec = self.after_hist[key].tolist()
        title_str += '%6s: %s  %s\n' % (key,
                                        ' '.join([str(x) for x in sem_vec]),
                                        self.symbol_cases[key])

      plt.title(title_str, family='monospace')

    plt.axis('off')
    plt.legend()
    plt.tight_layout()

    if fname:
      plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    else:
      plt.show()

    plt.close()
