'''
File    :   preprocessing.py
Note    :   Pre-processing file for NeuroBio 881.
            This file mainly slices fMRI data on time axis and make corresponding emotion labels.
Time    :   2023/04/13 23:01
Author  :   Kevin Xiu
Version :   1.0
Contact :   xiuyanming@gmail.com
'''

import nibabel as nib
import numpy as np
import torch
import os


def main():

    # ! requires modification if you want to precess new data
    nii_path = 'D:\\Kevin2023-2\\NeuroBio881\Project\\archive'
    sub_folder = '\\Sub-02\\'  # !
    base_file_name = 'wrsub-02_task-emotionalfaces_run-5_bold.nii'  # !

    save_dir = 'D:\\Kevin2023-2\\NeuroBio881\\Project\\emotion_fMRI\\sub2\\'  # !

    file = nii_path + sub_folder + base_file_name

    # 1. 提取niifile文件（其实是提取文件）
    img = nib.load(file)
    # 获取niifile数据
    img_fdata = img.get_fdata(dtype=np.float64)

    print(img_fdata.shape)
    print(type(img_fdata))
    print(img_fdata[:, :, :, 0].shape)
    print(img.affine)

    scrambled_start = [0, 90]  # !
    happy_start = [30, 120]
    sad_start = [15, 105]
    angry_start = [60, 150]
    neutral_start = [45, 135]
    [scrambled_counter, happy_counter, sad_counter,
        angry_counter, neutral_counter] = [0] * 5

    duration = 15

    final_img = nib.Nifti1Image(img_fdata[:, :, :, 80], img.affine)  # !
    nib.save(final_img, os.path.join(save_dir+'base.nii.gz'))

    for i in range(185):

        final_img = nib.Nifti1Image(img_fdata[:, :, :, i], img.affine)
        emotion = ''

        if (i >= scrambled_start[0] and i < scrambled_start[0]+duration) or (i >= scrambled_start[1] and i < scrambled_start[1]+duration):
            emotion = 'scrambled'
            scrambled_counter += 1
            # ! requires modification if you want to precess new data
            file_name = "sub02_run05_" + emotion + "_" + \
                str(1000+scrambled_counter) + ".nii.gz"  # ! here
            nib.save(final_img, os.path.join(save_dir+file_name))
        elif (i >= happy_start[0] and i < happy_start[0]+duration) or (i >= happy_start[1] and i < happy_start[1]+duration):
            emotion = 'happy'
            happy_counter += 1
            file_name = "sub02_run05_" + emotion + "_" + \
                str(1000+happy_counter) + ".nii.gz"
            nib.save(final_img, os.path.join(save_dir+file_name))
        elif (i >= sad_start[0] and i < sad_start[0]+duration) or (i >= sad_start[1] and i < sad_start[1]+duration):
            emotion = 'sad'
            sad_counter += 1
            file_name = "sub02_run05_" + emotion + "_" + \
                str(1000+sad_counter) + ".nii.gz"
            nib.save(final_img, os.path.join(save_dir+file_name))
        elif (i >= angry_start[0] and i < angry_start[0]+duration) or (i >= angry_start[1] and i < angry_start[1]+duration):
            emotion = 'angry'
            angry_counter += 1
            file_name = "sub02_run05_" + emotion + "_" + \
                str(1000+angry_counter) + ".nii.gz"
            nib.save(final_img, os.path.join(save_dir+file_name))
        elif (i >= neutral_start[0] and i < neutral_start[0]+duration) or (i >= neutral_start[1] and i < neutral_start[1]+duration):
            emotion = 'neutral'
            neutral_counter += 1
            file_name = "sub02_run05_" + emotion + "_" + \
                str(1000+neutral_counter) + ".nii.gz"
            nib.save(final_img, os.path.join(save_dir+file_name))

        print(f'{i} finished, 185 in total')

    pass


if __name__ == '__main__':
    main()
