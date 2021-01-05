
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
        self.pair_n = 5000
        self.action_limit = 0.1
        self.fengine = Fdata()
        self.gen_data()

    def gen_data(self):
        theta = 0*math.pi
        self.rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]]).astype(np.float)

        self.matrix = np.array([[2, 0, 0],
                                [0, 2, 0],
                                [0, 0, 1]]).astype(np.float)
        self.matrix = self.matrix.dot(self.rotation)

        self.dataA = np.random.random((self.pair_n,3))
        self.dataA[:,-1] = 1
        self.dataB = (self.matrix.dot(self.dataA.T)).T
        self.dataA = self.dataA[:,:2]
        self.dataB = self.dataB[:,:2]
        self.action = np.random.random((self.pair_n,2))

    def __getitem__(self, item):
        itemA = self.dataA[item]
        id = random.sample(range(self.pair_n),1)[0]
        itemB = self.dataB[id]
        return itemA,(itemB,self.action[id],itemB), self.dataB[item]

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
        self.pi = math.pi

    def __getitem__(self, item):
        theta1 = 1/3.*math.pi
        action = (random.random()*2-1)*self.action_limit
        theta2 = theta1+action*2*math.pi
        point1 = self.theta2point(theta1)
        point2 = self.theta2point(theta2)
        return point1,action,point2

    def theta2point(self,theta):
        if (theta>=0 and theta<self.pi/2) or theta>self.pi/6*11:
            x = math.cos(theta)/(2*math.sin(theta+self.pi/3))
            y = math.sin(theta)/(2*math.sin(theta+self.pi/3))
        elif theta>=self.pi/2 and theta<self.pi*7/6:
            x = - math.cos(self.pi-theta) / (2 * math.sin(self.pi-theta + self.pi / 3))
            y = math.sin(self.pi-theta) / (2 * math.sin(self.pi-theta + self.pi / 3))
        else:
            x = - math.tan(self.pi*3/2-theta)/2
            y = - 0.5
        return np.array([x/2-2,y/2])

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

