import glob
import numpy as np
import vectorize

from tqdm import tqdm
from skimage import measure

GA_ROOT = ''
PREPROCESS_ROOT = GA_ROOT + 'preprocess/'
SPLITS_ROOT = GA_ROOT + 'splits/x_val/'


def get_ins_sem(ins_id, ins_full, sem_full):
  ins_mask = (ins_full == ins_id)
  sem_mask = sem_full[ins_mask]
  unique, counts = np.unique(sem_mask, return_counts=True)
  ins_sem = unique[np.argmax(counts)]

  return ins_sem


semantic_fs = glob.glob(PREPROCESS_ROOT + 'semantic/*.npy')
fp_ids = [x.split('/')[-1].split('.')[0] for x in semantic_fs]
assert len(fp_ids) == 200

num_corners = 0
num_edges = 0
num_regions = 0

num_regions_by_sem = {
  1: 0,
  2: 0,
  3: 0,
  4: 0,
  5: 0,
  6: 0,
  7: 0
}
sem_names = ['bg', 'outer', 'inner', 'window', 'door', 'portal', 'room', 'frame']

heights = []
widths = []

for fp_id in tqdm(fp_ids):
  sem_full = np.load(PREPROCESS_ROOT + 'semantic/%s.npy' % fp_id)
  ins_full = np.load(PREPROCESS_ROOT + 'instance/%s.npy' % fp_id)
  graph = vectorize.get_graph(sem_full)

  heights.append(sem_full.shape[0])
  widths.append(sem_full.shape[1])

  num_corners += len(graph.nodes)
  num_edges += len(graph.edges)

  for ins_id in np.unique(ins_full):
    ins_sem = get_ins_sem(ins_id, ins_full, sem_full)

    if ins_sem == 0:
      continue

    num_regions += 1
    num_regions_by_sem[ins_sem] += 1


print('totals')
print('  corners: %d' % num_corners)
print('  edges  : %d' % num_edges)
print('  regions: %d' % num_regions)

for (sem, count) in num_regions_by_sem.items():
  print('  %7s: %d' % (sem_names[sem], count))
print('')

print('averages')
print('  height : %f' % (sum(heights) / len(semantic_fs)))
print('  width  : %f' % (sum(widths) / len(semantic_fs)))
print('  corners: %f' % (num_corners / len(semantic_fs)))
print('  edges  : %f' % (num_edges / len(semantic_fs)))
print('  regions: %f' % (num_regions / len(semantic_fs)))

for (sem, count) in num_regions_by_sem.items():
  print('  %7s: %f' % (sem_names[sem], count / len(semantic_fs)))
