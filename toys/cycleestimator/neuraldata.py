


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



class CDFdata(Data.Dataset):
    def __init__(self,opt=None):
        super(CDFdata,self).__init__()
        self.train = opt.istrain
        self.test_id = opt.test_id
        self.root_path = os.path.join(opt.data_root,'test{}'.format(self.test_id))
        self.trans = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.imgA,self.imgB,self.seqid = [],[],[]
        self.stateA,self.stateB,self.action = [],[],[]
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
            state_file = os.path.join(self.root_path,file)
            epoch_stateA,epoch_stateB,epoch_action = self.get_state(state_file)
            self.stateA.extend(epoch_stateA)
            self.stateB.extend(epoch_stateB)
            self.action.extend(epoch_action)

            img_dir = os.path.join(self.root_path,'epoch-{}'.format(epoch_id))
            epoch_imgA,epoch_imgB = self.get_img(img_dir)
            self.imgA.extend(epoch_imgA)
            self.imgB.extend(epoch_imgB)

        self.sample_num = len(self.imgA)
        assert (self.sample_num == len(self.imgB))
        assert (self.sample_num == len(self.stateA))
        assert (self.sample_num == len(self.stateB))
        assert (self.sample_num == len(self.action))
        self.stateA = np.array(self.stateA)
        self.stateB = np.array(self.stateB)
        self.action = np.array(self.action)


    def get_img(self,dir):
        img_list = os.listdir(dir)
        img_list = sorted(img_list,key=lambda x:int(x.split('.')[0]))
        img_list = [os.path.join(dir,x) for x in img_list]
        return img_list[:-2],img_list[1:-1]

    def get_state(self,file):
        state_sequence, action_sequence = [],[]
        sequence_id = int(file.split('-')[-1].split('.')[0])
        state_limit = 0.5
        action_limit = 0.05
        for line in open(file,'r').readlines():
            line = line.strip('\n')
            if 'now_pos' in line:
                state = line.split(':')[1].split()
                state = [float(x)/state_limit for x in state][:3]
                state_sequence.append(state)
            if 'action' in line:
                action = line.split(':')[1].split()
                action = [float(x)/action_limit for x in action][:3]
                action_sequence.append(action)
        return state_sequence[:-1],state_sequence[1:],action_sequence[:-1]

    def get_state_sample(self):
        id = random.sample(range(self.sample_num), 1)[0]
        return self.stateA[id],self.action[id],self.stateB[id]

    def get_img_sample(self):
        id = random.sample(range(self.sample_num), 1)[0]
        img1 = self.trans(Image.open(self.imgA[id]))
        img2 = self.trans(Image.open(self.imgB[id]))
        return (img1,self.action[id],img2),(self.stateA[id],self.stateB[id])

    def __getitem__(self, item):
        item1,imgstate = self.get_img_sample()
        item2 = self.get_state_sample()
        return item1,item2,imgstate

    def __len__(self):
        return self.sample_num

    @classmethod
    def get_loader(cls,opt):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size=opt.batch_size,
            shuffle=True
        )
