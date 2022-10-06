import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from matplotlib.colors import ListedColormap


GA_ROOT = ''
PREPROCESS_ROOT = GA_ROOT + 'preprocess/'
SPLITS_ROOT = GA_ROOT + 'splits/x_val/'


my_colors = {
  'bg':     (255, 255, 255, 255),  # white
  'outer':  (223, 132, 224, 255),  # purple
  'inner':  ( 84, 135, 255, 255),  # blue
  'window': (255, 170,  84, 255),  # orange
  'door':   (101, 255,  84, 255),  # green
  'frame':  (255, 246,  84, 255),  # yellow
  'room':   (230, 230, 230, 255),  # gray
  'symbol': (255,  87,  87, 255),  # red
}
colors = np.array(list(my_colors.values())) / 255.
my_cmap = ListedColormap(colors)


aoi_fs = glob.glob(PREPROCESS_ROOT + 'refined/*.txt')
fp_ids = [x.split('/')[-1].split('.')[0] for x in aoi_fs]

for fp_id in tqdm(fp_ids):
  fp_img = Image.open(PREPROCESS_ROOT + 'fp_img/%s.jpg' % fp_id)
  pred_img = Image.open(PREPROCESS_ROOT + 'semantic_pred/%s.png' % fp_id)
  r_img = Image.open(PREPROCESS_ROOT + 'refined/%s.png' % fp_id)
  rh_img = Image.open(PREPROCESS_ROOT + 'refined_heuristic/%s.png' % fp_id)
  gt_img = Image.open(PREPROCESS_ROOT + 'semantic/%s.png' % fp_id)

  with open(PREPROCESS_ROOT + 'refined/%s.txt' % fp_id, 'r') as f:
    f.readline()  # skip header line
    lines = [line.strip().split(',') for line in f]

  for i, (fp_id, need_refine, before, after, mini, minj, maxi, maxj) in enumerate(lines):
    if int(need_refine) and (not int(before)) and int(after):
      box = (int(minj), int(mini), int(maxj), int(maxi))

      fp_crop = fp_img.crop(box)
      fp_crop = fp_crop.resize((400, 400))

      pred_crop = pred_img.crop(box)
      pred_crop = pred_crop.resize((400, 400))

      r_crop = r_img.crop(box)
      r_crop = r_crop.resize((400, 400))

      rh_crop = rh_img.crop(box)
      rh_crop = rh_crop.resize((400, 400))

      gt_crop = gt_img.crop(box)
      gt_crop = gt_crop.resize((400, 400))

      fp_crop.save('./qual_vis/%s_%02d_1_fp.png' % (fp_id, i))
      pred_crop.save('./qual_vis/%s_%02d_1_pred.png' % (fp_id, i))
      r_crop.save('./qual_vis/%s_%02d_1_r.png' % (fp_id, i))
      rh_crop.save('./qual_vis/%s_%02d_1_rh.png' % (fp_id, i))
      gt_crop.save('./qual_vis/%s_%02d_1_gt.png' % (fp_id, i))
