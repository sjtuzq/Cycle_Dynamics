
"""
This file provides models
including Generator and discriminator

"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class GModel(nn.Module):
    def __init__(self,opt):
        super(GModel,self).__init__()
        self.opt = opt
        self.stn = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        self.biasfc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )

    def forward2(self, action):
        feature = torch.ones_like(action).float().cuda()
        trans = self.stn(feature).view(-1, 2, 2)
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 1]).astype(np.float32))).\
                         view(1, 4).repeat(action.shape[0], 1).view(- 1, 2,2).cuda()
        trans = trans + iden

        action = torch.bmm(trans,action.unsqueeze(-1)).squeeze()

        bias = self.biasfc(feature)

        return action+bias


    def forward1(self, data):
        return self.fc(data)

    def forward0(self, data):
        theta = 1/4.*math.pi
        self.rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]]).astype(np.float)

        self.matrix = np.array([[2, 0, 4],
                                [0, 2, 1],
                                [0, 0, 1]]).astype(np.float)
        self.matrix = self.matrix.dot(self.rotation)

        return data

    def forward(self, data):
        return self.forward1(data)


class DModel(nn.Module):
    def __init__(self,opt):
        super(DModel,self).__init__()
        self.opt = opt
        self.fc = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )

    def forward(self, data):
        return self.fc(data)


if __name__ == '__main__':
    model = GModel().cuda()
    data = torch.rand(32,64,64).cuda()
    output = model(data)
    print(output.shape)

    model = DModel().cuda()
    data = torch.rand(32,64,64).cuda()
    output = model(data)
    print(output.shape)