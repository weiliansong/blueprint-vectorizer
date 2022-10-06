import os
import torch
import json
import numpy as np
import skimage
import skimage.transform
import random
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils.config import Struct, load_config, compose_config_str


class FloorplanDataset(Dataset):
    def __init__(self, phase, test_fold_id, configs):
        super(FloorplanDataset, self).__init__()

        if phase not in ['train', 'test', 'val']:
            raise ValueError('invalid phase {}'.format(phase))

        self.phase = phase
        self.test_fold_id = test_fold_id
        self.configs = configs
        self.augmentation = configs.augmentation
        self.final_size = configs.final_size
        self.method = configs.method
        self.pad_size = configs.pad_size
        self.margin_size = configs.margin_size

        # depending on the phase, load up the corresponding splits
        f_ids = []
        split_root = os.path.join(configs.base_dir, configs.split_dir)

        if phase == 'train':
          for fold_id in range(10):
            if fold_id != test_fold_id:
              with open(os.path.join(split_root, 'ids_%d.json' % fold_id), 'r') as f:
                f_ids.extend(json.load(f))

          assert len(f_ids) == 180

        elif phase in ['test', 'val']:
          with open(os.path.join(split_root, 'ids_%d.json' % test_fold_id), 'r') as f:
            f_ids.extend(json.load(f))

          assert len(f_ids) == 20

        else:
          raise Exception('Unknown phase')

        self.images = {}
        self.labels = {}

        print('Caching %d floorplans' % len(f_ids))
        for f_id in f_ids:
          img_dir = os.path.join(self.configs.base_dir, self.configs.image_dir, "{}.jpg".format(f_id))
          label_dir = os.path.join(self.configs.base_dir, self.configs.mask_instance_dir, "{}.npy".format(f_id))

          image = Image.open(img_dir)
          image = image.convert('RGB')
          image = self.normalize(image)

          label = np.load(label_dir)
          
          self.images[f_id] = image
          self.labels[f_id] = label

        self.floorplan_ids = f_ids * configs.num_crop
        np.random.shuffle(self.floorplan_ids)

    def __len__(self):
        return len(self.floorplan_ids)

    def __getitem__(self, idx):
        np.random.seed()
        random.seed()
        torch.cuda.manual_seed_all(0)
       

        floorplan_id = self.floorplan_ids[idx]
        image = self.images[floorplan_id]
        label = self.labels[floorplan_id]

        # resizing
        ratio = np.random.uniform(low=0.9, high=1.1)
        h = int(self.final_size * ratio)
        w = int(self.final_size * ratio)
        image, label = self.random_crop(image, label, h, w, self.final_size, self.pad_size, self.margin_size)
  

        if self.augmentation:
            image, label = self.transform(image, label)

        image = self.normalize(np.asarray(image))
        label = np.asarray(label)

        image = torch.Tensor(image)
        image = image.permute([2,0,1])

        label = torch.Tensor(label)

        data = {
            'image': image,
            'label': label,    
        }

        return data

    def normalize(self, image):
        image = np.array(image, dtype=np.float32) / 255.0
        return image

    def transform(self, image, label):

        # rotate 0 or 90 or 180 or 270 degree
        angle = np.random.randint(3)  # clock-wise rotation
        angle = angle * 90
        # angle = np.random.uniform(low=0, high=360, size=(1,))  # clock-wise rotation
        # angle = angle // 10 * 10  # make the degree a multiplier of 10
        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle, fill=(0,))
        
        # apply flip
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)


        return image, label


    def full_crop(self, image, label, h, w, pad_size):
        '''
        utility function to resize sample(PIL image and label) to a given dimension
        without cropping information. the network takes in tensors with dimensions
        that are multiples of 32.

        :param img: numpy image to resize
        :param label: numpy array with the label to resize
        :param h: desired height
        :param w: desired width

        :return: the resized image, label
        '''

        img_h, img_w, _= image.shape
        
        # crop floorplan image to get rid of extra space around the floorplan
        miny, minx, maxy, maxx = self.get_region_bbox(label)
        # add small padding
        miny = max(miny-pad_size, 0)
        minx = max(minx-pad_size, 0)
        maxy = min(maxy+pad_size, img_h)
        maxx = min(maxx+pad_size, img_w)

        image = image[miny:maxy, minx:maxx, :]
        label = label[miny:maxy, minx:maxx]

        img_h, img_w, _= image.shape

        # print(img_h, img_w)

        # add padding to become a square
        image = np.pad(image, [((max(img_h, img_w)-img_h)//2, (max(img_h, img_w)-img_h)//2), \
                               ((max(img_h, img_w)-img_w)//2, (max(img_h, img_w)-img_w)//2), (0, 0)], \
                                mode='constant', constant_values=1.0)

        label = np.pad(label, [((max(img_h, img_w)-img_h)//2, (max(img_h, img_w)-img_h)//2), \
                               ((max(img_h, img_w)-img_w)//2, (max(img_h, img_w)-img_w)//2)], \
                                mode='constant', constant_values=0.0)

        # resizing
        scale_size = (h, w)
        image = Image.fromarray((image*255).astype(np.uint8))
        label = Image.fromarray((label).astype(np.uint8))

        image = image.resize(scale_size, Image.ANTIALIAS)
        label = label.resize(scale_size, Image.ANTIALIAS)

        image = self.normalize(np.asarray(image))
        label = np.asarray(label)

        # image_padded = resize(image_padded, [self.crop_size, self.crop_size])
        # label_padded = resize(label_padded,
        #                   [self.crop_size, self.crop_size],
        #                   order=0,
        #                   preserve_range=True,
        #                   anti_aliasing=False)

        return image, label


    def random_crop(self, image, label, h, w, final_size, pad_size, margin_size):
        '''
        utility function to resize sample(PIL image and label) to a given dimension
        without cropping information. the network takes in tensors with dimensions
        that are multiples of 32.

        :param img: numpy image to resize
        :param label: numpy array with the label to resize
        :param h: desired height
        :param w: desired width

        :return: the resized image, label
        '''

        # NOTE not doing margin cropping anymore, as input is already cropped
        # img_h, img_w, _= image.shape
        # # ''' -------------------------------------- 
        # # crop floorplan image to get rid of extra space around the floorplan
        # miny, minx, maxy, maxx = self.get_region_bbox(label)
        # # add small padding
        # miny = max(miny-margin_size, 0)
        # minx = max(minx-margin_size, 0)
        # maxy = min(maxy+margin_size, img_h)
        # maxx = min(maxx+margin_size, img_w)

        # image = image[miny:maxy, minx:maxx, :]
        # label = label[miny:maxy, minx:maxx]
	    # ''' ---------------------------------------------
        pad_size = pad_size + margin_size
        image = np.pad(image, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode='edge')
        label = np.pad(label, [(pad_size, pad_size), (pad_size, pad_size)], mode='edge')
	
        img_h, img_w, _= image.shape

        minx = np.random.choice(range(0, img_w-w))
        miny = np.random.choice(range(0, img_h-h))
        maxx = minx + w
        maxy = miny + h
        # print(minx, miny)

        image = image[miny:maxy, minx:maxx, :]
        label = label[miny:maxy, minx:maxx]
	
        # resizing
        scale_size = (final_size, final_size)
        
        # image = Image.fromarray((image*255).astype(np.uint8))
        # label = Image.fromarray((label).astype(np.uint8))

        # image = image.resize(scale_size, Image.ANTIALIAS)
        #label = label.resize(scale_size, Image.ANTIALIAS)
        image = skimage.transform.resize(image, scale_size)
        label = skimage.transform.resize(label,
                        scale_size,
                        mode='edge',
                        anti_aliasing=False,
                        anti_aliasing_sigma=None,
                        preserve_range=True,
                        order=0)
        
        image = Image.fromarray((image*255).astype(np.uint8))
        label = Image.fromarray((label).astype(np.uint8))

        # image = self.normalize(np.asarray(image))
        # label = np.asarray(label)

        return image, label

    def get_region_bbox(self, region_mask):
        yy, xx = np.nonzero(region_mask > 0)

        minx = int(np.min(xx))
        miny = int(np.min(yy))
        maxx = int(np.max(xx))
        maxy = int(np.max(yy))

        return [miny, minx, maxy, maxx]


def loader_test(configs):
  from tqdm import tqdm

  train_dataset = FloorplanDataset('train', 0, configs=configs)

  train_loader = DataLoader(train_dataset,
                            batch_size=configs.batch_size,
                            num_workers=configs.num_workers,
                            shuffle=True)
  
  for iter_i, batch_data in enumerate(tqdm(train_loader)):
    continue


# check dataloder ...
if __name__ == '__main__':

    config_dict = load_config(file_path='utils/config.yaml')
    configs = Struct(**config_dict)

    loader_test(configs)
    exit(0)

    train_dataset = FloorplanDataset('test', 0, configs = configs) 
    
    for idx, batch_data in enumerate(train_dataset):
        continue

        image = batch_data['image']
        label = batch_data['label']

        image = image.permute([1,2,0])

        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150)

        # don't visualize background as a 0 label
       # label = np.ma.masked_where(label == 0, label)

        ax1.imshow(image)
        ax1.set_axis_off()

        ax2.imshow(image)
        ax2.imshow(label, cmap='nipy_spectral', alpha=0.7)
        ax2.set_axis_off()

        plt.tight_layout()
        plt.show()
        plt.close()
        #  print(wait)



# img = Image.fromarray((image*255).astype(np.uint8))
# img.show()
