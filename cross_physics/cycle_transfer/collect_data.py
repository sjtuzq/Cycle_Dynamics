
import os
import gym
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from models import Actor,Forwardmodel,Inversemodel

def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


class TD3(object):
    def __init__(self,policy_path,state_dim,action_dim,max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.weight_path = policy_path
        self.actor.load_state_dict(torch.load(self.weight_path))
        print('policy weight loaded!')
        # self.axmodel = Axmodel(opt).cuda()
        # self.ax_weight_path = opt.axmodel_path
        # self.axmodel.load_state_dict(torch.load(self.ax_weight_path))
        # print('axmodel weight loaded!')

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def online_action(self,state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.axmodel(state,self.actor(state)).cpu().data.numpy().flatten()
        return action

    def online_axmodel(self,state,axmodel):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = axmodel(state,self.actor(state)).cpu().data.numpy().flatten()
        return action


class CycleData:
    def __init__(self, opt):
        self.pair_n = 3000
        self.opt = opt
        self.env = gym.make(opt.env)
        self.env.seed(0)
        random.seed(0)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.log_root = opt.log_root
        self.episode_n = opt.episode_n
        self.policy_path = os.path.join(opt.log_root,
                        '{}_base/models/TD3_{}_0_actor'.format(opt.env,opt.env))
        self.policy = TD3(self.policy_path,self.state_dim,self.action_dim,self.max_action)
        self.data1 = self.collect(self.opt.data_id)
        print('----------- Dataset initialized ---------------')
        print('-----------------------------------------------')
        self.sample_n1 = self.data1[0].shape[0]
        self.reset()

        opt.state_dim = self.state_dim
        opt.action_dim = self.action_dim
        self.model = Forwardmodel(opt).cuda()
        self.train_forward()
        print('-----------------------------------------------')
        self.inverse_model = Inversemodel(opt).cuda()
        self.train_inverse()
        print('-----------------------------------------------\n')

    def sample(self, batch_size=32):
        id1 = random.sample(range(self.sample_n1), batch_size)
        sample1 = (self.to_device(self.data1[0][id1]),
                   self.to_device(self.data1[1][id1]),
                   self.to_device(self.data1[2][id1]))
        return sample1

    def reset(self):
        idx = list(range(self.sample_n1))
        random.shuffle(idx)
        now = self.data1[0][idx]
        act = self.data1[1][idx]
        nxt = self.data1[2][idx]
        self.data1 = (now,act,nxt)
        self.pos = 0

    def sample1(self,batch_size=32):
        start = self.pos*batch_size
        end = start+batch_size
        if end>=self.sample_n1:
            start = 0
            end = start+batch_size
        self.pos = end
        id1 = list(range(start,end))
        sample1 = (self.to_device(self.data1[0][id1]),
                   self.to_device(self.data1[1][id1]),
                   self.to_device(self.data1[2][id1]))
        return sample1

    def to_device(self,data):
        return torch.tensor(data).float().cuda()

    def collect(self, data_id):
        self.env_logs = safe_path(os.path.join(self.log_root,'{}_data'.format(self.opt.env)))
        data_folder = safe_path(os.path.join(self.env_logs,'{}_{}'.format(self.opt.data_type,self.opt.data_id)))
        self.data_folder = data_folder
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        now_path = os.path.join(data_folder,'now.npy')
        nxt_path = os.path.join(data_folder,'nxt.npy')
        act_path = os.path.join(data_folder,'act.npy')
        try:
            now_obs = np.load(now_path)
            nxt_obs = np.load(nxt_path)
            action = np.load(act_path)
            if not self.opt.force:
                return (now_obs, action, nxt_obs)
        except:
            print('start to create data')

        now_buffer, action_buffer, nxt_buffer = [], [], []
        episode_r = 0.
        self.env.reset()
        for episode in tqdm(range(self.episode_n)):
            now_obs, action, nxt_obs = [], [], []
            obs, done = self.env.reset(), False
            done = False
            while not done:
                now_obs.append(obs)
                # offline policy sample
                act = self.policy.select_action(obs)
                # act = self.env.action_space.sample()
                # online policy sample
                # act = self.policy.online_action(obs)
                new_obs, r, done, info = self.env.step(act)
                action.append(act)
                nxt_obs.append(new_obs)
                obs = new_obs
                episode_r += r
            now_buffer.extend(now_obs)
            action_buffer.extend(action)
            nxt_buffer.extend(nxt_obs)
        print('average reward: {:.2f}'.format(episode_r/self.episode_n))
        now_obs = np.stack(now_buffer)
        action = np.stack(action_buffer)
        nxt_obs = np.stack(nxt_buffer)

        np.save(now_path, now_obs)
        np.save(act_path, action)
        np.save(nxt_path, nxt_obs)

        return (now_obs, action, nxt_obs)

    def train_forward(self):
        self.weight_path = os.path.join(self.data_folder, 'forward.pth')
        try:
            self.model.load_state_dict(torch.load(self.weight_path))
            print('load forward model correctly!')
            return 0
        except:
            print('start to train forward model!')
        optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)
        loss_fn = nn.L1Loss()

        for epoch in range(30):
            epoch_loss, cmp_loss = 0, 0
            if epoch==10:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
            elif epoch==20:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            for i in (range(self.pair_n)):
                item = self.sample()
                state, action, result = item
                out = self.model(state, action)
                loss = loss_fn(out, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print('epoch:{} loss:{:.7f}'.format(epoch, epoch_loss / self.pair_n))
            torch.save(self.model.state_dict(), self.weight_path)

        epoch_loss, cmp_loss = 0, 0
        for i in (range(self.pair_n)):
            item = self.sample()
            state, action, result = item
            out = self.model(state, action)
            loss = loss_fn(out, result)
            epoch_loss += loss.item()
        print('evaluation loss:{:.7f}'.format(epoch_loss / self.pair_n))

    def train_inverse(self):
        self.inverse_weight_path = os.path.join(self.data_folder, 'inverse.pth')
        try:
            self.inverse_model.load_state_dict(torch.load(self.inverse_weight_path))
            print('load inverse model correctly!')
            return 0
        except:
            print('start to train inverse model!')
        optimizer = torch.optim.Adam(self.inverse_model.parameters(),lr=1e-3)
        loss_fn = nn.L1Loss()

        for epoch in range(30):
            epoch_loss, cmp_loss = 0, 0
            if epoch==10:
                optimizer = torch.optim.Adam(self.inverse_model.parameters(), lr=3e-4)
            elif epoch==20:
                optimizer = torch.optim.Adam(self.inverse_model.parameters(), lr=1e-4)
            for i in (range(self.pair_n)):
                item = self.sample()
                state, action, result = item
                out = self.inverse_model(state, result)
                loss = loss_fn(out, action)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print('epoch:{} loss:{:.7f}'.format(epoch, epoch_loss / self.pair_n))
            torch.save(self.inverse_model.state_dict(), self.inverse_weight_path)

        epoch_loss, cmp_loss = 0, 0
        for i in (range(self.pair_n)):
            item = self.sample()
            state, action, result = item
            out = self.inverse_model(state, result)
            loss = loss_fn(out, action)
            epoch_loss += loss.item()
        print('evaluation loss:{:.7f}'.format(epoch_loss / self.pair_n))


    def online_test(self,axmodel,episode_n=100,imgpath=None):
        save_flag = False
        if imgpath is not None:
            if not os.path.exists(imgpath):
                os.mkdir(imgpath)
            save_flag = True
        with torch.no_grad():
            now_buffer, action_buffer, nxt_buffer = [], [], []
            reward_buffer = []
            for episode in (range(episode_n)):
                now_obs, action, nxt_obs = [], [], []
                obs = self.env.reset()
                done = False
                episode_r = 0.
                if save_flag:
                    episode_path = os.path.join(imgpath, 'episode_{}'.format(episode))
                    if not os.path.exists(episode_path):
                        os.mkdir(episode_path)
                count = 0
                while not done:
                    img = self.env.sim.render(mode='offscreen', camera_name='track', width=256, height=256)
                    if save_flag:
                        Image.fromarray(img[::-1, :, :]).save(os.path.join(episode_path, 'img_{}.jpg'.format(count)))
                        count += 1
                        print(episode,count)

                    now_obs.append(obs)
                    act = self.policy.online_axmodel(obs,axmodel)
                    new_obs, r, done, info = self.env.step(act)
                    action.append(act)
                    nxt_obs.append(new_obs)
                    obs = new_obs
                    episode_r += r
                reward_buffer.append(episode_r)
                now_buffer.extend(now_obs)
                action_buffer.extend(action)
                nxt_buffer.extend(nxt_obs)
                # print(episode_r/(episode+1))
            episode_r = sum(reward_buffer)
            # print('average reward: {:.2f}'.format(episode_r/episode_n))
            return np.array(reward_buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control dataset analyzer')
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--force", type=bool, default=True)
    parser.add_argument("--log_root", default="../../../logs/cross_physics")
    parser.add_argument('--data_type', type=str, default='arma3', help='data type')
    parser.add_argument('--data_id', type=int, default=2, help='data id')
    parser.add_argument('--episode_n', type=int, default=100, help='episode number')
    opt = parser.parse_args()

    dataset = CycleData(opt)
    item = dataset.sample()
    now,act,nxt = item
    print(now.shape,act.shape,nxt.shape)
    print(now.mean().item(),act.mean().item(),nxt.mean().item())


