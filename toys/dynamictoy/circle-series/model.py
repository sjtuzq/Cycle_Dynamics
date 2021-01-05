
"""
This file provides models
including Generator and discriminator

"""


import torch
import torch.nn as nn



class GModel(nn.Module):
    def __init__(self,opt):
        super(GModel,self).__init__()
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