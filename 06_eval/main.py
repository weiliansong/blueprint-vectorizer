import glob
import argparse

import numpy as np

from tqdm import tqdm

import vectorize
from metric import Metric


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


def get_recall_and_precision(tp, fp, length):
    recall = tp / (length + 1e-8)
    precision = tp / (tp+fp+1e-8)
    return recall, precision


def compute_f_score(precision, recall):
  return 2.0 * precision * recall / (recall + precision + 1e-8)


GA_ROOT = ''
PREPROCESS_ROOT = GA_ROOT + 'preprocess/'
SPLITS_ROOT = GA_ROOT + 'splits/x_val/'

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', dest='input_folder',
                    action='store', type=str)
args = parser.parse_args()
print('PR for %s' % args.input_folder)

semantic_fs = glob.glob(PREPROCESS_ROOT + '%s/*.npy' % args.input_folder)
fp_ids = [x.split('/')[-1].split('.')[0] for x in semantic_fs]

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
  sem_pred = np.load(PREPROCESS_ROOT + '%s/%s.npy' % (args.input_folder, fp_id))

  sem_gt[sem_gt == 24] = 7

  gt_data = get_data(sem_gt)
  pred_data = get_data(sem_pred)

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

with open(args.input_folder.split('/')[0] + '_gt.csv', 'w') as f:
  for fp_id, gt_id, conv_id in all_gt_mappings:
    f.write('%s,%d,%d\n' % (fp_id, gt_id, conv_id))

with open(args.input_folder.split('/')[0] + '_pred.csv', 'w') as f:
  for fp_id, conv_id, gt_id in all_pred_mappings:
    f.write('%s,%d,%d\n' % (fp_id, conv_id, gt_id))

with open(args.input_folder.split('/')[0] + '.txt', 'w') as f:
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
