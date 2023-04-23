'''
File    :   train.py
Note    :   Train a nn model on emotion-fMRI data to make predictions
Time    :   2023/04/13 23:47
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

# Options and Global Variables
# modify this main_dir ifyou are working on another PC
main_dir = 'D:\\Kevin2023-2\\NeuroBio881\\Project\\'
data_dir = main_dir + 'emotion_fMRI\\train_sub2'
total_data_number = 750
training_data_number = 500
batch_size = 100
epochs = 100
checkpoints = main_dir + 'ckpts'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = True
debug = False


def main():
    train_set = create_dataset(data_dir)

    # Split the training and testing sets
    # train_set, test_set = torch.utils.data.random_split(
    #     emotion_dataset, [training_data_number, test_data_number])

    # Dataloader: Automatically switched numpy data to Tensor!
    train_loader = create_dataloader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    # Define a NN model and its optimizer
    my_nn = simpleNN_2D().double()
    my_nn.to(device)
    optimizer = optim.SGD(my_nn.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    # record loss and accuracy
    loss_list, acc_list = [], []
    best_acc = 0

    if train == True:
        # Define a timer
        start = time.time()

        # Start Training
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}:')
            # Set the training mode
            my_nn.train()

            epoch_loss = 0.0
            correct = 0.0
            total_examples = 0

            for batch_idx, (img_batch, label_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                img_batch = img_batch.double().to(device)
                label_batch = label_batch.to(device)
                label_batch = torch.tensor(label_batch, dtype=torch.long)
                outputs = my_nn(img_batch)

                # outputs should have x samples and y types, so shape = x*y;
                # label_batch should have x samples so shape = x
                loss = criterion(outputs, label_batch)
                loss.backward()

                dummy_max, pred = torch.max(outputs, 1)
                pred = torch.unsqueeze(pred, 0)

                optimizer.step()

                correct += pred.eq(label_batch).sum()
                total_examples += label_batch.size(0)
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            avg_acc = correct / total_examples
            loss_list.append(avg_loss)
            acc_list.append(avg_acc.item())

            print('[epoch %d] loss: %.5f accuracy: %.4f time: %.4fs' %
                  (epoch + 1, avg_loss, avg_acc, time.time()-start))
            if best_acc <= avg_acc:
                best_acc = avg_acc
                print('Saving best acc model...')
                torch.save(my_nn.state_dict(), os.path.join(
                    checkpoints, 'best_acc.pth'))
            if epoch % 10 == 0:
                print(f'Saving epoch # {epoch} model...')
                torch.save(my_nn.state_dict(), os.path.join(
                    checkpoints, f'epoch{epoch}.pth'))

        # Save loss and accuracy
        save_file = np.array([loss_list, acc_list]).T
        frame = pd.DataFrame(save_file, index=range(1, epochs+1),
                             columns=['loss', 'acc'])
        frame.to_csv('epochs_vs_loss_and_acc.csv', index=',')
        print('Saving final model...')
        torch.save(my_nn.state_dict(), os.path.join(
            checkpoints, 'final.pth'))

    if debug == True:
        # Debugging
        for img_batch, label_batch in train_loader:
            # It's important to reassign! See https://stackoverflow.com/questions/60563115/.
            img_batch = img_batch.double().to(device)
            print(img_batch)
            print(label_batch)
            print(img_batch.shape)
            print(label_batch.shape)
            print(my_nn(img_batch).shape)
            break

    pass


if __name__ == '__main__':
    main()
