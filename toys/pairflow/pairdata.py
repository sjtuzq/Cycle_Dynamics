


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


class Fdata(Data.Dataset):
    def __init__(self,opt=None):
        super(Fdata,self).__init__()
        self.train = opt.istrain
        self.test_id = opt.test_id
        self.root_path = os.path.join(opt.data_root,'test{}'.format(self.test_id))
        self.trans = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.imgA,self.imgB,self.action,self.seqid = [],[],[],[]
        epoch_list = []
        for file in os.listdir(self.root_path):
            if 'txt' in file:
                epoch_list.append(file)
        epoch_n = len(epoch_list)
        if self.train:
            epoch_list = epoch_list[:int(epoch_n*0.8)]
        else:
            epoch_list = epoch_list[int(epoch_n*0.8):]
        for file in epoch_list:
            epoch_id = int(file.split('.')[0].split('-')[1])
            action_file = os.path.join(self.root_path,file)

            img_dir = os.path.join(self.root_path,'epoch-{}'.format(epoch_id))
            epoch_imgA,_ = self.get_img(img_dir)

            second_img_dir = os.path.join(self.root_path,'second-epoch-{}'.format(epoch_id))
            _, epoch_imgB = self.get_img(second_img_dir)

            self.imgA.extend(epoch_imgA)
            self.imgB.extend(epoch_imgB)
        assert (len(self.imgA)==len(self.imgB))

    def get_img(self,dir):
        img_list = os.listdir(dir)
        img_list = sorted(img_list,key=lambda x:int(x.split('.')[0]))
        img_list = [os.path.join(dir,x) for x in img_list]
        return img_list[:-1],img_list[1:]

    def __getitem__(self, item):
        img1 = self.trans(Image.open(self.imgA[item]))
        img2 = self.trans(Image.open(self.imgB[item]))
        return img1,img2

    def __len__(self):
        return len(self.imgA)

    @classmethod
    def get_loader(cls,opt):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size=opt.batch_size,
            shuffle=True
        )
