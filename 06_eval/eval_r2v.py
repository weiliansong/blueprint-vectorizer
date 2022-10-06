import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import vectorize
from metric import Metric


GA_ROOT = ''
PREPROCESS_ROOT = GA_ROOT + 'preprocess/'
SPLITS_ROOT = GA_ROOT + 'splits/x_val/'


def plot_graph(data):
  plt.imshow(data['segmentation'] / 7., cmap='nipy_spectral')

  for (a,b) in data['edges']:
    a_i, a_j = data['corners'][a]
    b_i, b_j = data['corners'][b]
    plt.plot([a_j-0.5, b_j-0.5], [a_i-0.5, b_i-0.5], linewidth=2, markersize=3)

  plt.axis('off')
  plt.tight_layout()
  plt.show()
  plt.close()


def get_data(segmentation):
  graph = vectorize.get_graph(segmentation)
  nodes = [graph.nodes[n]['ij'] for n in sorted(graph.nodes)]
  edges = list(graph.edges)

  data = {
    'corners': np.array(nodes),
    'edges': np.array(edges),
    'segmentation': segmentation
  }
  return data


def get_r2v_data(fp_id):
  segmentation = np.load(GA_ROOT + 'r2v_results/all/%s_result_line.npy' % fp_id)
  segmentation = segmentation.astype(np.int64)

  # double-check that we have the same number of corners and edges
  with open(GA_ROOT + 'r2v_results/all/%s_corners.txt' % fp_id, 'r') as f:
    f.readline()
    num_corners = int(f.readline().strip())
    num_edges = 0

    for line in f:
      tokens = line.strip().split('\t')
      if '.' not in tokens[0]:
        num_edges += 1

  return get_data(segmentation)


def get_recall_and_precision(tp, fp, length):
    recall = tp / (length + 1e-8)
    precision = tp / (tp+fp+1e-8)
    return recall, precision


def compute_f_score(precision, recall):
  return 2.0 * precision * recall / (recall + precision + 1e-8)


print('PR for r2v')

semantic_fs = glob.glob(GA_ROOT + 'r2v_results/all/*_result_line.npy')
fp_ids = [x.split('/')[-1][:14] for x in semantic_fs]
assert len(fp_ids) == 200

metric = Metric()

corner_tp = 0.0
corner_fp = 0.0
corner_length = 0.0
edge_tp = 0.0
edge_fp = 0.0
edge_length = 0.0
region_tp = 0.0
region_fp = 0.0
region_length = 0.0

region_sem_tp = {}
region_sem_fp = {}
region_sem_length = {}

for sem in range(8):
  region_sem_tp[sem] = 0.0
  region_sem_fp[sem] = 0.0
  region_sem_length[sem] = 0.0

all_gt_mappings = []
all_pred_mappings = []

for idx, fp_id in enumerate(fp_ids):
  print('%d / %d' % (idx, len(fp_ids)))

  sem_gt = np.load(PREPROCESS_ROOT + 'semantic/%s.npy' % fp_id)

  gt_data = get_data(sem_gt)
  pred_data = get_r2v_data(fp_id)

  score = metric.calc(gt_data, pred_data)

  corner_tp += score['corner_tp']
  corner_fp += score['corner_fp']
  corner_length += score['corner_length']
  edge_tp += score['edge_tp']
  edge_fp += score['edge_fp']
  edge_length += score['edge_length']
  region_tp += score['region_tp']
  region_fp += score['region_fp']
  region_length += score['region_length']

  for sem in range(8):
    region_sem_tp[sem] += score['region_sem_tp'][sem]
    region_sem_fp[sem] += score['region_sem_fp'][sem]
    region_sem_length[sem] += score['region_sem_length'][sem]

  for gt_id, conv_id in score['gt_mappings'].items():
    all_gt_mappings.append((fp_id, gt_id, conv_id))

  for conv_id, gt_id in score['pred_mappings'].items():
    all_pred_mappings.append((fp_id, conv_id, gt_id))

with open('r2v_gt.csv', 'w') as f:
  for fp_id, gt_id, conv_id in all_gt_mappings:
    f.write('%s,%d,%d\n' % (fp_id, gt_id, conv_id))

with open('r2v_pred.csv', 'w') as f:
  for fp_id, conv_id, gt_id in all_pred_mappings:
    f.write('%s,%d,%d\n' % (fp_id, conv_id, gt_id))

with open('r2v.txt', 'w') as f:
  # corner
  corner_recall, corner_precision = get_recall_and_precision(corner_tp,
                                                             corner_fp,
                                                             corner_length)
  corner_f_score = compute_f_score(corner_precision, corner_recall)

  print('')
  print('corners:')
  print('  precision: %.3f' % corner_precision)
  print('     recall: %.3f' % corner_recall)
  print('    f_score: %.3f' % corner_f_score)
  print('')

  f.write('corners - precision: %.3f recall: %.3f f_score: %.3f\n' \
            % (corner_precision, corner_recall, corner_f_score))

  # edge
  edge_recall, edge_precision = get_recall_and_precision(edge_tp,
                                                         edge_fp,
                                                         edge_length)
  edge_f_score = compute_f_score(edge_precision, edge_recall)

  print('edges:')
  print('  precision: %.3f' % edge_precision)
  print('     recall: %.3f' % edge_recall)
  print('    f_score: %.3f' % edge_f_score)
  print('')

  f.write('edges - precision: %.3f recall: %.3f f_score: %.3f\n' \
            % (edge_precision, edge_recall, edge_f_score))

  # region
  region_recall, region_precision = get_recall_and_precision(region_tp,
                                                             region_fp,
                                                             region_length)
  region_f_score = compute_f_score(region_precision, region_recall)

  print('regions:')
  print('  precision: %.3f' % region_precision)
  print('     recall: %.3f' % region_recall)
  print('    f_score: %.3f' % region_f_score)
  print('')

  f.write('regions - precision: %.3f recall: %.3f f_score: %.3f\n' \
            % (region_precision, region_recall, region_f_score))

  # region by sem
  for sem in range(7, 8):
    region_sem_recall, region_sem_precision = get_recall_and_precision(region_sem_tp[sem],
                                                                       region_sem_fp[sem],
                                                                       region_sem_length[sem])
    region_sem_f_score = compute_f_score(region_sem_precision, region_sem_recall)

    print('region %d:' % sem)
    print('  precision: %.3f' % region_sem_precision)
    print('     recall: %.3f' % region_sem_recall)
    print('    f_score: %.3f' % region_sem_f_score)
    print('')

    f.write('region %d - precision: %.3f recall: %.3f f_score: %.3f\n' \
              % (sem, region_sem_precision, region_sem_recall, region_sem_f_score))
