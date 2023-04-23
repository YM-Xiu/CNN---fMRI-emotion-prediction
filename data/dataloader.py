'''
File    :   dataloader.py
Note    :
Time    :   2023/04/16 18:17
Author  :   Kevin Xiu
Version :   1.0
Contact :   xiuyanming@gmail.com
'''

from torch.utils.data.dataloader import DataLoader


def create_dataloader(dataset, batch_size=10, shuffle=True):
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader


def main():

    pass


if __name__ == '__main__':
    main()
