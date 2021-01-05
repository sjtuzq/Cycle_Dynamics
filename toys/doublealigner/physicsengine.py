

import os
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from PIL import Image


class Engine(object):
    def __init__(self,test_id=1,mass=1.0):
        self.mass = mass
        self.xy_limit = 1
        self.action_limit = 0.2
        self.pos = np.array([0.,0.])
        self.data_path = os.path.join('./tmp/data/test{}'.format(test_id))
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        self.img_path = os.path.join(self.data_path,'imgs')
        if not os.path.exists(self.img_path):
            os.mkdir(self.img_path)

    def step(self,action):
        action = action/self.mass
        self.pos = (self.pos+action).clip(min=0,max=self.xy_limit)
        return self.pos

    def sample_action(self):
        action = (np.random.random(2) - 0.5) * 2 * self.action_limit*self.mass
        return action

    def render_data(self,num=10):
        state_buffer = []
        action_buffer = []
        last_pos = env.pos
        state_buffer.append(last_pos)
        img_path = os.path.join(self.img_path,'0.jpg')
        self.plot_circle(last_pos,img_path)
        for i in tqdm(range(num)):
            action = env.sample_action()
            pos = env.step(action)
            img_path = os.path.join(self.img_path,'{}.jpg'.format(i+1))
            self.plot_circle(pos,img_path)
            action_buffer.append(action)
            state_buffer.append(pos)
            last_pos = pos
        state_buffer = np.array(state_buffer)
        action_buffer = np.array(action_buffer)
        np.save(os.path.join(self.data_path,'state.npy'),state_buffer)
        np.save(os.path.join(self.data_path,'action.npy'),action_buffer)

    def plot_circle(self,pos=None,img_path=None):
        plt.rcParams['figure.figsize'] = (4.0,4.0)
        fig, ax = plt.subplots()
        ax.set_xlim(-0.2,1.2)
        ax.set_ylim(-0.2,1.2)
        patches = [Circle((pos[0],pos[1]),0.1)]
        colors = 100*np.random.rand(len(patches))
        p = PatchCollection(patches, alpha=0.4)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        plt.axis('off')
        # plt.show()
        plt.savefig(img_path)
        plt.cla()
        plt.clf()



class CDFdata(Data.Dataset):
    def __init__(self,opt=None):
        super(CDFdata,self).__init__()
        self.train = opt.istrain
        self.test_id1 = opt.test_id1
        self.test_id2 = opt.test_id2
        self.root_path1 = os.path.join(opt.data_root,'test{}'.format(self.test_id1))
        self.root_path2 = os.path.join(opt.data_root,'test{}'.format(self.test_id2))
        self.trans = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        # self.stateA,self.stateB,self.action1 = self.get_state()
        # self.imgA,self.imgB,self.action2 = self.get_img()
        self.img1A,self.img1B,self.state1A,self.state1B,self.action1 = self.get_data(1)
        self.img2A, self.img2B, self.state2A, self.state2B, self.action2 = self.get_data(2)

        self.sample_num = len(self.img1A)
        assert (self.sample_num == len(self.img1B))
        assert (self.sample_num == self.state2A.shape[0])
        assert (self.sample_num == self.state2B.shape[0])
        assert (self.sample_num == self.action1.shape[0])
        assert (self.sample_num == self.action2.shape[0])

    def get_data(self,id):
        if id==1:
            path = self.root_path1
        else:
            path = self.root_path2
        img_path = os.path.join(path,'imgs')
        img_list = os.listdir(img_path)
        img_list = sorted(img_list,key=lambda x:int(x.split('.')[0]))
        img_list = [os.path.join(img_path,x) for x in img_list]
        state_file = os.path.join(path, 'state.npy')
        action_file = os.path.join(path, 'action.npy')
        state = np.load(state_file)
        action = np.load(action_file)
        return img_list[:-1],img_list[1:],state[:-1],state[1:],action


    def get_state_sample(self):
        id = random.sample(range(self.sample_num), 1)[0]
        return self.state1A[id],self.action1[id],self.state1B[id]


    def get_img_sample(self):
        id = random.sample(range(self.sample_num), 1)[0]
        img_now = self.trans(Image.open(self.img2A[id]))
        img_nxt = self.trans(Image.open(self.img2B[id]))
        return (img_now,self.action2[id],img_nxt),(self.state2A[id],self.state2B[id])


    def __getitem__(self, item):
        item1,img_gt_state = self.get_img_sample()
        item2 = self.get_state_sample()
        return item1,item2,img_gt_state


    def __len__(self):
        return self.sample_num


    @classmethod
    def get_loader(cls,opt):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size=opt.batch_size,
            shuffle=opt.istrain,
            num_workers=8,
        )


if __name__ == '__main__':
    # env = Engine(test_id=1,mass=1)
    # env.render_data(1000)
    # env = Engine(test_id=2,mass=1)
    # env.render_data(1000)

    # env = Engine(test_id=3, mass=0.8)
    # env.render_data(1000)
    # env = Engine(test_id=4, mass=0.6)
    # env.render_data(1000)

    # env = Engine(test_id=5, mass=1.2)
    # env.render_data(1000)
    # env = Engine(test_id=6, mass=1.4)
    # env.render_data(1000)

    env = Engine(test_id=7,mass=1)
    env.render_data(5000)

    env = Engine(test_id=8,mass=0.6)
    env.render_data(5000)

