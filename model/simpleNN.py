'''
File    :   simpleNN.py
Note    :
Time    :   2023/04/16 18:47
Author  :   Kevin Xiu
Version :   1.0
Contact :   xiuyanming@gmail.com
'''
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class simpleNN_3D(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv3d(1, 3, (3, 3, 3), stride=(1, 1, 1))
        # self.fc1 = nn.Linear()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 2, (5, 5, 5), stride=(2, 2, 2)),
            # nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(2, 4, (5, 5, 5), stride=(2, 2, 2)),
            # nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(4, 8, (5, 5, 5), stride=(2, 2, 2)),
            # nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8*7*9*7, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
    pass


# class simpleNN_1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.conv1 = nn.Conv3d(1, 3, (3, 3, 3), stride=(1, 1, 1))
#         # self.fc1 = nn.Linear()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 8, (3, 3), stride=(2, 2)),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, (3, 3), stride=(2, 2)),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, (3, 3), stride=(2, 2)),
#             # nn.BatchNorm3d(16),
#             nn.ReLU()
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(32*3*3, 128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.ReLU(),
#             nn.Linear(32, 5)
#         )

#     def forward(self, x):
#         out = self.conv_layers(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc_layers(out)
#         return out
#     pass


class simpleNN_2D(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv3d(1, 3, (3, 3, 3), stride=(1, 1, 1))
        # self.fc1 = nn.Linear()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, (5, 5), stride=(2, 2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), stride=(2, 2)),
            # nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32*8*10, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            # nn.ReLU(),
            # nn.Linear(32, 5)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
    pass


def main():

    pass


if __name__ == '__main__':
    main()
