
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

class Fdata():
    def __init__(self,test_id=1,opt=None):
        super(Fdata,self).__init__()
        self.train = opt.istrain
        self.test_id = test_id
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
            epoch_imgA,epoch_imgB = self.get_img(img_dir)
            epoch_action,each_seqid = self.get_action(action_file)
            self.imgA.extend(epoch_imgA)
            self.imgB.extend(epoch_imgB)
            self.action.extend(epoch_action)
            self.seqid.extend(each_seqid)

        assert (len(self.imgA)==len(self.action))
        assert (len(self.imgB)==len(self.action))
        assert (len(self.seqid)==len(self.action))


    def get_action(self,file):
        action_sequence, sequence_idlist = [],[]
        sequence_id = int(file.split('-')[-1].split('.')[0])
        action_limit = 0.05
        for line in open(file,'r').readlines():
            line = line.strip('\n')
            if 'action' in line:
                action = line.split(':')[1].split()
                action = [float(x)/action_limit for x in action][:3]
                action_sequence.append(action)
                sequence_idlist.append(sequence_id)
        return action_sequence,sequence_idlist


    def get_img(self,dir):
        img_list = os.listdir(dir)
        img_list = sorted(img_list,key=lambda x:int(x.split('.')[0]))
        img_list = [os.path.join(dir,x) for x in img_list]
        return img_list[:-1],img_list[1:]


class Mydata(Data.Dataset):
    def __init__(self,opt=None):
        self.imgA_id, self.imgB_id = opt.imgA_id,opt.imgB_id
        self.imgA = Fdata(self.imgA_id,opt).imgA
        self.imgB = Fdata(self.imgB_id,opt).imgB
        self.imgA_sample_id = list(range(len(self.imgA)))
        self.imgB_sample_id = list(range(len(self.imgB)))
        self.trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, item):
        itemA = random.sample(self.imgA_sample_id,1)[0]
        itemB = random.sample(self.imgB_sample_id,1)[0]
        imgA = self.trans(Image.open(self.imgA[itemA]))
        imgB = self.trans(Image.open(self.imgB[itemB]))
        return imgA,imgB

    def __len__(self):
        return min(len(self.imgA),len(self.imgB))

    @classmethod
    def get_loader(cls,opt):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size = opt.batch_size,
            shuffle = opt.istrain,
            num_workers = 8,
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--train', type=bool, default=True, help='whether training or test')
    parser.add_argument('--batch_size', type=int, default=8, help='experiment type')
    parser.add_argument('--exp_id', type=int, default=1, help='experiment type')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--obj_size', type=int, default=5, help='image size')
    parser.add_argument('--domain', type=str, default='whole', help='data distribution domain')
    opt = parser.parse_args()
    dataset = Mydata.get_loader(opt)

    for i,item in enumerate(dataset):
        print(i,item[0].shape,item[1].shape)

