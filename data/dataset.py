'''
File    :   dataloader.py
Note    :
Time    :   2023/04/16 18:06
Author  :   Kevin Xiu
Version :   1.0
Contact :   xiuyanming@gmail.com
'''

import os
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import shutil
import glob
import random
from data.preprocess import preprocess

from PIL import Image


patch_size_x = 48
patch_size_y = 48
patch_size_z = 48

# The base scans. Subjects show no emotion here.

blank_list = [[], []]  # 2 subjects * 5 runs

for j in range(2):
    for i in range(5):

        blank = nib.load(
            f'D:\\Kevin2023-2\\NeuroBio881\\Project\\emotion_fMRI\\base_sub{j+1}_run{i+1}.nii.gz')
        blank_data = blank.get_fdata(dtype=np.float64)[:, :, 40]
        blank_list[j].append(blank_data)

# blank = nib.load(
#     'D:\\Kevin2023-2\\NeuroBio881\\Project\\emotion_fMRI\\base_sub1_run2.nii.gz')
# blank_data_s1_r2 = blank.get_fdata(dtype=np.float64)[:, :, 40]

# blank = nib.load(
#     'D:\\Kevin2023-2\\NeuroBio881\\Project\\emotion_fMRI\\base_sub1_run3.nii.gz')
# blank_data_s1_r3 = blank.get_fdata(dtype=np.float64)[:, :, 40]

# blank = nib.load(
#     'D:\\Kevin2023-2\\NeuroBio881\\Project\\emotion_fMRI\\base_sub1_run4.nii.gz')
# blank_data_s1_r4 = blank.get_fdata(dtype=np.float64)[:, :, 40]

# blank = nib.load(
#     'D:\\Kevin2023-2\\NeuroBio881\\Project\\emotion_fMRI\\base_sub1_run5.nii.gz')
# blank_data_s1_r5 = blank.get_fdata(dtype=np.float64)[:, :, 40]


class EmotionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_path = os.listdir(self.data_dir)
        self.label_dict = {'happy': 0, 'sad': 1,
                           'scrambled': 2, 'neutral': 3, 'angry': 4}

    def __getitem__(self, index):
        img_name = self.img_path[index]
        image_item_path = os.path.join(self.data_dir, img_name)

        img = nib.load(image_item_path)
        img_fdata = img.get_fdata(dtype=np.float64)
        # img_fdata = np.zeros((79, 95, 79))
        # img_fdata = img_fdata * (1/img_fdata.max())

        # make labels
        for e, n in self.label_dict.items():
            if e in img_name:
                # img_fdata = preprocess(img_fdata, e) - blank_data_s1_r1
                label = n

                break

        # apply blank masks
        for j in range(2):
            for i in range(5):
                if ('run0'+str(i+1)) in img_name and ('sub0'+str(j+1)) in img_name:
                    img_fdata = preprocess(img_fdata, e) - blank_list[j][i]
                    break

        # img_patch = img_fdata.reshape(1, 79, 95, 79)
        img_patch = img_fdata.reshape(1, 79, 95)

        return img_patch, label

    def __len__(self):
        return len(self.img_path)


# class EmotionDataset_1(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.img_path = os.listdir(self.data_dir)
#         self.label_dict = {'horse': 0, 'bird': 1,
#                            'automobile': 2, 'airplane': 3, 'dog': 4}

#     def __getitem__(self, index):
#         img_name = self.img_path[index]
#         image_item_path = os.path.join(self.data_dir, img_name)
#         img = Image.open(image_item_path)
#         img_fdata = np.array(img.getdata())

#         # make labels
#         for e, n in self.label_dict.items():
#             if e in img_name:
#                 label = n
#                 break

#         img_patch = img_fdata.reshape(3, 32, 32)

#         return img_patch, label

#     def __len__(self):
#         return len(self.img_path)


def create_dataset(data_dir):
    dataset = EmotionDataset(data_dir)
    return dataset


def create_dataset_1(data_dir):
    dataset = EmotionDataset_1(data_dir)
    return dataset


def main():
    data_dir = 'D:\\Kevin2023-2\\NeuroBio881\\Project\\emotion_fMRI'
    data = create_dataset(data_dir)
    print(len(data))
    img_1, label_1 = data[1]
    print(img_1.shape)
    print(label_1)
    pass


if __name__ == '__main__':
    main()
