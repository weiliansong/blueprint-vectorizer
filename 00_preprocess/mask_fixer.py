# So here's the problem with the masks. I annotate wall and openings, and
# room masks are obtained by finding enclosed regions (ignoring small gaps
# between walls and openings). Yes that stuff in the parentheses is the
# problem! This script fixes those small black bits in the annotation (both
# instance and semantic by filling them in through dilation.

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from skimage.morphology import dilation, erosion, square


def fill_long_gaps(fp_id, image, mask, k=3, ratio=1):
  # 1. find the background of floorplan through dilation and erosion
  pos_mask = mask.copy()
  pos_mask[pos_mask > 0] = 1

  pos_mask = dilation(pos_mask, square(k))
  pos_mask = erosion(pos_mask, square(k))

  bg_mask = np.logical_not(pos_mask).astype(int)

  # this is to check that we indeed have the perimeter of the floorplan if
  # there are any black bits left, we would have more than two cc components
  components = measure.label(bg_mask)

  if len(np.unique(components)) != 2:
    plt.imshow(image, cmap='gray')
    plt.imshow(bg_mask, cmap='gray', alpha=0.8)
    plt.axis('off')
    plt.title(fp_id)
    plt.tight_layout()
    plt.show()

  # 2. find all the black bits of the floorplan
  bits_mask = (mask == 0).astype(np.int64)
  bits_mask[bg_mask == 1] = 0

  if False:
    plt.imshow(image, cmap='gray')
    plt.imshow(bits_mask, cmap='gray', alpha=0.8)
    plt.axis('off')
    plt.show()

  # 3. only do something to the tall and skinny bits
  bits_mask_copy = np.copy(bits_mask).astype(np.float32)
  bits_mask = measure.label(bits_mask)

  for bit_id in np.unique(bits_mask):
    if bit_id == 0:
      continue

    bit_mask = (bits_mask == bit_id)

    ii, jj = np.nonzero(bit_mask)
    mini = ii.min()
    minj = jj.min()
    maxi = ii.max()+1
    maxj = jj.max()+1

    # the bit is long height-wise
    if (maxi-mini) / (maxj-minj) >= ratio:
      candidates, count = np.unique(mask[mini:maxi, minj-1:maxj+1],
                                    return_counts=True)

    # the bit is long width-wise
    elif (maxj-minj) / (maxi-mini) >= ratio:
      candidates, count = np.unique(mask[mini-1:maxi+1, minj:maxj],
                                    return_counts=True)

    else:
      # indicate that this region is not fixed
      bits_mask_copy[bit_mask] = 0.5
      continue

    # get the 2nd most common ID
    pixel_counts = []

    for candidate_id in candidates:
      if candidate_id == 0:
        continue

      pixel_count = (mask == candidate_id).sum()
      pixel_counts.append((pixel_count, candidate_id))

    if len(pixel_counts) == 1:
      # print(fp_id)
      # plt.imshow(image, cmap='gray')
      # plt.imshow(bit_mask, cmap='hot', alpha=0.5)
      # plt.show()
      replace_id = pixel_counts[0][1]

    else:
      replace_id = sorted(pixel_counts)[::-1][1][1]

    mask[bit_mask] = replace_id

  if False:
    bits_mask_copy = np.ma.masked_where(bits_mask_copy < 0.4, bits_mask_copy)
    random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))
    fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)

    ax1.imshow(image, cmap='gray')
    ax1.imshow(bits_mask_copy, cmap='winter')
    ax1.set_axis_off()

    ax2.imshow(mask, cmap=random_cmap)
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()
    plt.close()

  # make sure we don't have any background bits left
  pos_mask = mask.copy()
  pos_mask[pos_mask > 0] = 1

  bg_mask = np.logical_not(pos_mask).astype(int)

  components = measure.label(bg_mask)
  assert len(np.unique(components)) == 2

  return mask


# 1. find the background of floorplan through dilation and erosion
# 2. find all the black bits of the floorplan
# 3. dilate all components of the mask to see what the bits' labels would be
# 4. assign labels to black bits, and paste labelled bits onto original mask
def remove_black_bits(image, mask, k=11):
  # 1. find the background of floorplan through dilation and erosion
  pos_mask = mask.copy()
  pos_mask[pos_mask > 0] = 1

  pos_mask = dilation(pos_mask, square(k))
  pos_mask = erosion(pos_mask, square(k))

  bg_mask = np.logical_not(pos_mask).astype(int)

  # this is to check that we indeed have the perimeter of the floorplan if
  # there are any black bits left, we would have more than two cc components
  components = measure.label(bg_mask)

  if len(np.unique(components)) != 2:
    plt.imshow(image, cmap='gray')
    plt.imshow(bg_mask, cmap='gray', alpha=0.8)
    plt.axis('off')
    plt.show()

  # 2. find all the black bits of the floorplan
  bits_mask = (mask == 0).astype(np.int64)
  bits_mask[bg_mask == 1] = 0

  if False:
    plt.imshow(image, cmap='gray')
    plt.imshow(bits_mask, cmap='gray', alpha=0.8)
    plt.axis('off')
    plt.show()

  # 3. dilate all components of the mask to see what the bits' labels would be
  overfilled_mask = dilation(mask, square(k))

  if False:
    _overfilled_mask = overfilled_mask.astype(np.float32)
    _overfilled_mask /= _overfilled_mask.max()

    plt.imshow(image, cmap='gray')
    plt.imshow(_overfilled_mask, cmap='nipy_spectral', alpha=0.8)
    plt.axis('off')
    plt.show()

  # 4. assign labels to black bits, and paste labelled bits onto original mask
  bits_labels = overfilled_mask.copy()
  bits_labels[bits_mask == 0] = 0

  fixed_mask = mask + bits_labels

  # make sure we didn't overlap any regions
  assert len(np.unique(fixed_mask)) == len(np.unique(mask))

  if False:
    _fixed_mask = fixed_mask.astype(np.float32)
    _fixed_mask /= _fixed_mask.max()

    plt.imshow(_fixed_mask, cmap='nipy_spectral')
    plt.axis('off')
    plt.show()

  # TODO there is another check we can make in regards to number of components

  return fixed_mask


# NOTE not working...
def remove_black_bits_2(mask, k=7):
  # 1. find the background of floorplan through dilation and erosion
  pos_mask = mask.copy()
  pos_mask[pos_mask > 0] = 1

  pos_mask = dilation(pos_mask, square(k))
  pos_mask = erosion(pos_mask, square(k))

  bg_mask = np.logical_not(pos_mask).astype(int)

  # this is to check that we indeed have the perimeter of the floorplan if
  # there are any black bits left, we would have more than two cc components
  components = measure.label(bg_mask)
  assert len(np.unique(components)) == 2

  if False:
    plt.imshow(bg_mask, cmap='gray')
    plt.axis('off')
    plt.show()

  # 2. dilate all components of the mask to fill in gaps
  overfilled_mask = dilation(mask, square(k))

  if False:
    _overfilled_mask = overfilled_mask.astype(np.float32)
    _overfilled_mask /= _overfilled_mask.max()

    plt.imshow(_overfilled_mask, cmap='nipy_spectral')
    plt.axis('off')
    plt.show()

  # 3. cookie-cut overfilled floorplan with background mask
  fixed_mask = overfilled_mask.copy()
  fixed_mask[bg_mask == 1] = 0

  assert len(np.unique(fixed_mask)) == len(np.unique(mask))

  if True:
    _fixed_mask = fixed_mask.astype(np.float32)
    _fixed_mask /= _fixed_mask.max()

    plt.imshow(_fixed_mask, cmap='nipy_spectral')
    plt.axis('off')
    plt.show()
