'''
File    :   test.py
Note    :   Test on a trained model and make predictions on test fMRI data
Time    :   2023/04/13 23:48
Author  :   Kevin Xiu
Version :   1.0
Contact :   xiuyanming@gmail.com
'''

import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from data.dataset import create_dataset
from data.dataloader import create_dataloader
from model.simpleNN import simpleNN_2D
import torch.optim as optim
from PIL import Image
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# modify this main_dir ifyou are working on another PC
main_dir = 'D:\\Kevin2023-2\\NeuroBio881\\Project\\'
data_dir = main_dir + 'emotion_fMRI\\test_sub2'
ckpts_name = main_dir + 'ckpts\\epoch50.pth'
total_data_number = 750
training_data_number = 100
test_data_number = 50
batch_size = 100


def main():
    test_set = create_dataset(data_dir)
    test_set = create_dataset(data_dir)
    test_loader = create_dataloader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    my_nn = simpleNN_2D().double()
    my_nn.to(device)
    my_nn.load_state_dict(torch.load(ckpts_name))
    my_nn.eval()
    correct_test = 0
    with torch.no_grad():
        for batch_idx, (img_batch, label_batch) in enumerate(test_loader):
            img_batch = img_batch.double().to(device)
            label_batch = label_batch.to(device)
            label_batch = torch.tensor(label_batch, dtype=torch.long)

            # pass the data into the network and store the output
            outputs = my_nn(img_batch)
            # Get the prediction from the output
            dummy_max, pred = torch.max(outputs, 1)
            # Calculate the correct number and add the number to correct_test
            correct_test += pred.eq(label_batch).sum()

    print('Accuracy on the test images: %.2f %%' %
          (100 * correct_test / len(test_set)))
    pass


if __name__ == '__main__':
    main()
