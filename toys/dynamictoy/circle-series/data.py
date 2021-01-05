
"""
This file provides dataloader

"""

import os
import math
import torch
import random
import argparse
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


class Mydata(Data.Dataset):
    def __init__(self):
        super(Mydata,self).__init__()
        self.pair_n = 1000
        self.cut_n = 16.0

    def get_domain_A(self):
        theta = random.random()
        theta = int(theta*self.cut_n)
        theta = theta*2*math.pi/self.cut_n
        x = math.cos(theta) - 2
        y = math.sin(theta)
        return np.array([x,y])

    def get_domain_B(self):
        theta = random.random() * 2 * math.pi
        x = math.cos(theta) + 2
        y = math.sin(theta)
        return np.array([x, y])

    def __getitem__(self, item):
        itemA = self.get_domain_A()
        itemB = self.get_domain_B()
        return itemA,itemB

    def __len__(self):
        return self.pair_n

    @classmethod
    def get_loader(cls,opt):
        dataset = cls()
        return Data.DataLoader(
            dataset = dataset,
            batch_size = opt.batch_size,
            shuffle = opt.istrain,
            num_workers = 8,
        )

class CDFdata(Data.Dataset):
    def __init__(self,opt=None):
        super(CDFdata,self).__init__()
        self.opt = opt
        self.pair_n = 1000
        self.cut_n = 16.0
        self.action_limit = 0.1

    def get_domain_A(self):
        theta = random.random() * 2 * math.pi
        x = math.cos(theta) - 2
        y = math.sin(theta)
        return np.array([x,y])

    def get_domain_B(self):
        theta1 = random.random() * 2 * math.pi
        action = (random.random() * 2 - 1) * self.action_limit
        if self.opt.clockwise:
            theta2 = theta1 + action * 2 * math.pi
        else:
            theta2 = theta1 - action * 2 * math.pi
        point1 = np.array([math.cos(theta1) + 2, math.sin(theta1)])
        point2 = np.array([math.cos(theta2) + 2, math.sin(theta2)])
        return point1,action,point2

    def __getitem__(self, item):
        itemA = self.get_domain_A()
        itemB = self.get_domain_B()
        return itemA,itemB

    def __len__(self):
        return self.pair_n

    @classmethod
    def get_loader(cls,opt):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size = opt.batch_size,
            shuffle = opt.istrain,
            num_workers = 8,
        )

class Fdata(Data.Dataset):
    def __init__(self):
        super(Fdata,self).__init__()
        self.pair_n = 10000
        self.action_limit = 0.1

    def __getitem__(self, item):
        theta1 = random.random()*2*math.pi
        action = (random.random()*2-1)*self.action_limit
        theta2 = theta1+action*2*math.pi
        point1 = np.array([math.cos(theta1)-2,math.sin(theta1)])
        point2 = np.array([math.cos(theta2)-2,math.sin(theta2)])
        return point1,action,point2

    def __len__(self):
        return self.pair_n

    @classmethod
    def get_loader(cls,batch_size=16):
        dataset = cls()
        return Data.DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 8,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--batch_size', type=int, default=8, help='experiment type')
    parser.add_argument('--exp_id', type=int, default=1, help='experiment type')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--obj_size', type=int, default=5, help='image size')
    parser.add_argument('--domain', type=str, default='whole', help='data distribution domain')
    opt = parser.parse_args()
    dataset = Mydata.get_loader(opt)

    for i,item in enumerate(dataset):
        itemA,itemB = item
        plt.scatter(itemA[:,0].data.numpy(),itemA[:,1].data.numpy())
        plt.scatter(itemB[:, 0].data.numpy(), itemB[:, 1].data.numpy())
    plt.show()

