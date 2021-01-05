

import os
import gym
import torch
import random
import argparse
import numpy as np
from PIL import Image
import torch.utils.data as Data
from torchvision import transforms


class Robotdata(Data.Dataset):
    def __init__(self,opt=None):
        self.opt = opt
        self.istrain = opt.istrain
        self.log_root = opt.log_root
        self.env_logs = os.path.join(self.log_root, '{}_data'.format(self.opt.env))
        self.data_root1 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type1, self.opt.data_id1))
        self.data_root2 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type2, self.opt.data_id2))
        self.trans = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.img1A,self.img1B,self.state1A,self.state1B,self.action1 = self.load_data(self.data_root1)
        self.img2A, self.img2B, self.state2A, self.state2B, self.action2 = self.load_data(self.data_root2)
        self.clip_range = opt.clip_range
        self.mean = self.state1A.mean(0)
        self.std = self.state1A.std(0)
        self.state1A = np.clip((self.state1A-self.mean)/self.std,-self.clip_range,self.clip_range)
        self.state1B = np.clip((self.state1B-self.mean)/self.std,-self.clip_range,self.clip_range)
        self.state2A = np.clip((self.state2A-self.mean)/self.std,-self.clip_range,self.clip_range)
        self.state2B = np.clip((self.state2B-self.mean)/self.std,-self.clip_range,self.clip_range)

        self.sample_num1 = len(self.img1A)
        self.sample_num2 = len(self.img1B)

    def load_data(self,data_root):
        img_path = os.path.join(data_root, 'imgs')
        now_state = np.load(os.path.join(data_root,'now_state.npy'))
        next_state = np.load(os.path.join(data_root,'next_state.npy'))
        action = np.load(os.path.join(data_root,'action.npy'))
        now_img,next_img = self.get_imgs(img_path)

        pair_n = now_state.shape[0]
        assert (pair_n==next_state.shape[0])
        assert (pair_n==action.shape[0])
        assert (pair_n==len(now_img))
        assert (pair_n==len(next_img))

        return now_img,next_img,now_state,next_state,action

    def get_imgs(self,img_path):
        episode_list = os.listdir(img_path)
        episode_list = sorted(episode_list,key=lambda x:int(x.split('-')[1]))
        now_imglist,next_imglist = [],[]
        for dir in episode_list:
            episode_path = os.path.join(img_path,dir)
            tmp = os.listdir(episode_path)
            tmp = sorted(tmp,key=lambda x:int(x.split('_')[-1].split('.')[0]))
            tmp = [os.path.join(episode_path,x) for x in tmp]
            now_imglist.extend(tmp[:-1])
            next_imglist.extend(tmp[1:])
        return now_imglist,next_imglist

    def get_state_sample(self):
        id = random.sample(range(self.sample_num1), 1)[0]
        return self.state1A[id],self.action1[id],self.state1B[id]

    def get_img_sample(self):
        id = random.sample(range(self.sample_num2), 1)[0]
        img_now = self.trans(Image.open(self.img2A[id]))
        img_nxt = self.trans(Image.open(self.img2B[id]))
        return (img_now,self.action2[id],img_nxt),(self.state2A[id],self.state2B[id])

    def __getitem__(self, item):
        if self.opt.istrain:
            item1,img_gt_state = self.get_img_sample()
            item2 = self.get_state_sample()
            return item1,item2,img_gt_state
        else:
            img_now = self.trans(Image.open(self.img2A[item]))
            img_nxt = self.trans(Image.open(self.img2B[item]))
            item1,img_gt_state = (img_now,self.action2[item],img_nxt),(self.state2A[item],self.state2B[item])
            item2 = (self.state1A[item],self.action1[item],self.state1B[item])
            return item1,item2,img_gt_state

    def __len__(self):
        return min(self.sample_num1,self.sample_num2)

    @classmethod
    def get_loader(cls,opt=None):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size = opt.batch_size,
            shuffle = opt.istrain,
            num_workers = 32
        )


class RobotStackdata(Data.Dataset):
    def __init__(self,opt=None):
        self.opt = opt
        self.istrain = opt.istrain
        self.log_root = opt.log_root
        self.env_logs = os.path.join(self.log_root, '{}_data'.format(self.opt.env))
        self.data_root1 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type1, self.opt.data_id1))
        self.data_root2 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type2, self.opt.data_id2))
        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.stack_n = opt.stack_n
        self.img_size = opt.img_size
        self.trans = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.trans_stack = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.img1A,self.img1B,self.state1A,self.state1B,self.action1 = self.load_data(self.data_root1)
        self.img2A, self.img2B, self.state2A, self.state2B, self.action2 = self.load_data(self.data_root2)
        self.sample_num1 = len(self.img1A)
        self.sample_num2 = len(self.img2A)

        self.mean = self.state1A.mean(0)
        self.std = self.state1A.std(0)
        self.state1A = (self.state1A-self.mean)/self.std
        self.state1B = (self.state1B-self.mean)/self.std
        self.state2A = (self.state2A-self.mean)/self.std
        self.state2B = (self.state2B-self.mean)/self.std

    def load_data(self,data_root):
        img_path = os.path.join(data_root, 'imgs')
        now_state = np.load(os.path.join(data_root,'now_state.npy'))[:,:self.state_dim]
        next_state = np.load(os.path.join(data_root,'next_state.npy'))[:,:self.state_dim]
        action = np.load(os.path.join(data_root,'action.npy'))
        now_img,next_img = self.get_imgs(img_path)

        pair_n = now_state.shape[0]
        assert (pair_n==next_state.shape[0])
        assert (pair_n==action.shape[0])
        assert (pair_n==len(now_img))
        assert (pair_n==len(next_img))

        return now_img,next_img,now_state,next_state,action

    def get_imgs(self,img_path):
        episode_list = os.listdir(img_path)
        episode_list = sorted(episode_list,key=lambda x:int(x.split('-')[1]))
        now_imglist,next_imglist = [],[]
        for dir in episode_list:
            episode_path = os.path.join(img_path,dir)
            tmp = os.listdir(episode_path)
            tmp = sorted(tmp,key=lambda x:int(x.split('_')[-1].split('.')[0]))
            tmp = [os.path.join(episode_path,x) for x in tmp]
            now_imglist.extend(tmp[:-1])
            next_imglist.extend(tmp[1:])
        return now_imglist,next_imglist

    def get_state_sample(self,sample_num,id=None):
        if id is None:
            id = random.sample(range(sample_num), 1)[0]
        return self.state1A[id],self.action1[id],self.state1B[id]

    def read_img(self,path):
        img = Image.open(path)
        img = transforms.ToTensor()(img)
        if self.opt.env == 'HalfCheetah-v2':
            img = img[:, 128:220, 36:220]
        # elif self.opt.env == "InvertedDoublePendulum-v2":
        #     img = img[:,92:156,54:202]
        # elif self.opt.env == "Reacher-v2":
        #     img = img[:,128:384,128:384]
        # elif self.opt.env == "Swimmer-v2":
        #     img = img[:,84:196,24:232]
        img = transforms.ToPILImage()(img)
        img = self.trans_stack(img)
        return img

    def get_img_sample(self,sample_num,id=None):
        if id is None:
            id = random.sample(range(sample_num-self.stack_n+1), 1)[0] + self.stack_n-1
        img_now,img_nxt = [],[]
        for i in range(self.stack_n):
            img_now.append(self.read_img(self.img2A[id-self.stack_n+1+i]))
            img_nxt.append(self.read_img(self.img2B[id-self.stack_n+1+i]))
        img_now = torch.stack(img_now,0)
        img_nxt = torch.stack(img_nxt,0)
        return (img_now,self.action2[id],img_nxt),(self.state2A[id],self.state2B[id])

    def __getitem__(self, item):
        if self.opt.istrain:
            item1,img_gt_state = self.get_img_sample(self.sample_num1)
            item2 = self.get_state_sample(self.sample_num2)
            return item1,item2,img_gt_state
        else:
            item1,img_gt_state = self.get_img_sample(self.sample_num1,item)
            item2 = self.get_state_sample(self.sample_num2,item)
            return item1,item2,img_gt_state

    def __len__(self):
        return min(self.sample_num1,self.sample_num2)

    @classmethod
    def get_loader(cls,opt=None):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size = opt.batch_size,
            shuffle = opt.istrain,
            num_workers = 32
        )


class RobotStackFdata(Data.Dataset):
    def __init__(self,opt=None):
        self.opt = opt
        self.istrain = opt.istrain
        self.log_root = opt.log_root
        self.env_logs = os.path.join(self.log_root, '{}_data'.format(self.opt.env))
        self.data_root1 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type1, self.opt.data_id1))
        self.data_root2 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type2, self.opt.data_id2))
        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.stack_n = opt.stack_n
        self.img_size = opt.img_size
        self.trans = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.trans_stack = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.img1A,self.img1B,self.state1A,self.state1B,self.action1 = self.load_data(self.data_root1)
        self.img2A, self.img2B, self.state2A, self.state2B, self.action2 = self.load_data(self.data_root2)

        self.mean = self.state1A.mean(0)
        self.std = self.state1A.std(0)
        self.state1A = (self.state1A-self.mean)/self.std
        self.state1B = (self.state1B-self.mean)/self.std
        self.state2A = (self.state2A-self.mean)/self.std
        self.state2B = (self.state2B-self.mean)/self.std

        split_n1 = int(len(self.img1A) *0.8)
        split_n2 = int(len(self.img2A) *0.8)
        if self.opt.istrain:
            self.img1A = self.img1A[:split_n1]
            self.img1B = self.img1B[:split_n1]
            self.state1A = self.state1A[:split_n1]
            self.state1B = self.state1B[:split_n1]
            self.action1 = self.action1[:split_n1]
            self.img2A = self.img2A[:split_n2]
            self.img2B = self.img2B[:split_n2]
            self.state2A = self.state2A[:split_n2]
            self.state2B = self.state2B[:split_n2]
            self.action2 = self.action2[:split_n2]
        else:
            self.img1A = self.img1A[split_n1:]
            self.img1B = self.img1B[split_n1:]
            self.state1A = self.state1A[split_n1:]
            self.state1B = self.state1B[split_n1:]
            self.action1 = self.action1[split_n1:]
            self.img2A = self.img2A[split_n2:]
            self.img2B = self.img2B[split_n2:]
            self.state2A = self.state2A[split_n2:]
            self.state2B = self.state2B[split_n2:]
            self.action2 = self.action2[split_n2:]

        self.sample_num1 = len(self.img1A)
        self.sample_num2 = len(self.img2A)

    def load_data(self,data_root):
        img_path = os.path.join(data_root, 'imgs')
        now_state = np.load(os.path.join(data_root,'now_state.npy'))[:,:self.state_dim]
        next_state = np.load(os.path.join(data_root,'next_state.npy'))[:,:self.state_dim]
        action = np.load(os.path.join(data_root,'action.npy'))
        now_img,next_img = self.get_imgs(img_path)

        pair_n = now_state.shape[0]
        assert (pair_n==next_state.shape[0])
        assert (pair_n==action.shape[0])
        assert (pair_n==len(now_img))
        assert (pair_n==len(next_img))

        return now_img,next_img,now_state,next_state,action

    def get_imgs(self,img_path):
        episode_list = os.listdir(img_path)
        episode_list = sorted(episode_list,key=lambda x:int(x.split('-')[1]))
        now_imglist,next_imglist = [],[]
        for dir in episode_list:
            episode_path = os.path.join(img_path,dir)
            tmp = os.listdir(episode_path)
            tmp = sorted(tmp,key=lambda x:int(x.split('_')[-1].split('.')[0]))
            tmp = [os.path.join(episode_path,x) for x in tmp]
            now_imglist.extend(tmp[:-1])
            next_imglist.extend(tmp[1:])
        return now_imglist,next_imglist

    def get_state_sample(self):
        sample_num = self.sample_num1
        id = random.sample(range(sample_num), 1)[0]
        return self.state1A[id],self.action1[id],self.state1B[id]

    def __getitem__(self, item):
        item2 = self.get_state_sample()
        return item2

    def __len__(self):
        # return self.sample_num1
        return 100000

    @classmethod
    def get_loader(cls,opt=None):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size = opt.batch_size,
            shuffle = opt.istrain,
            num_workers = 8
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_id1', default=1, type=int)
    parser.add_argument('--test_id2', default=2, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--istrain', default=True, type=int)
    parser.add_argument('--save_img', default=True, type=int)
    parser.add_argument('--episode_n', default=1000, type=int)
    parser.add_argument('--horizon_n', default=100, type=int)
    parser.add_argument('--data_root', default='../../logs', type=str)


    opt = parser.parse_args()
    dataset = Robotdata.get_loader(opt)
    for i,item in enumerate(dataset):
        item1,item2,gt = item
        print(item1[0].shape)
