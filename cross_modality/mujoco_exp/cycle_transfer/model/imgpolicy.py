
import os
import gym
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class TD3(object):
    def __init__(self,policy_path,state_dim,action_dim,max_action,opt):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.weight_path = policy_path
        self.actor.load_state_dict(torch.load(self.weight_path))
        print('policy weight loaded!')
        self.stack_agent = Stackimg(opt)
        env_logs = os.path.join(opt.log_root, '{}_data'.format(opt.env))
        data_path = os.path.join(env_logs, '{}_{}'.format(opt.data_type1, opt.data_id1))
        self.mean_std_path = os.path.join(data_path,'now_state.npy')
        self.data = np.load(self.mean_std_path)
        self.mean = torch.tensor(self.data.mean(0)).float().cuda()
        self.std = torch.tensor(self.data.std(0)).float().cuda()
        self.clip_range = 5

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def select_action_from_img(self,img,imgmodel,goal):
        img = self.stack_agent.push(img[::-1, :, :])
        output = imgmodel(img.unsqueeze(0)).squeeze()
        pred_state = self.mean.clone()
        pred_state[:output.shape[0]] = output
        pred_state = pred_state * self.std + self.mean
        pred_state = pred_state.cpu().data.numpy()
        pred_state[2:4] = goal
        state = torch.FloatTensor(pred_state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action,pred_state


class Stackimg:
    def __init__(self,opt):
        self.stack_n = opt.stack_n
        self.img_size = opt.img_size
        self.env = opt.env
        self.trans_stack = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.buffer = []

    def push(self,img):
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        # if self.env == 'HalfCheetah-v2':
        #     img = img[:, 128:220, 36:220]
        # elif self.env == "InvertedDoublePendulum-v2":
        #     img = img[:, 92:156, 54:202]
        # elif self.env == "Reacher-v2":
        #     img = img[:, 128:384, 128:384]
        img = transforms.ToPILImage()(img)
        img = self.trans_stack(img)

        if len(self.buffer)<self.stack_n:
            self.buffer = [img]*self.stack_n
        else:
            self.buffer = self.buffer[1:]
            self.buffer.append(img)
        return torch.stack(self.buffer,0).float().cuda()


class ImgPolicy:
    def __init__(self, opt):
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
        try:
            self.policy = TD3(self.policy_path,self.state_dim,self.action_dim,self.max_action,opt)
            print(self.policy_path)
        except:
            print('no pre-trained policy model weight!')

    def online_test(self,imgmodel,episode_n=100):
        with torch.no_grad():
            gt,pred = [],[]
            reward_buffer = []

            for episode in tqdm(range(episode_n)):
                obs = self.env.reset()
                done = False
                episode_r = 0.
                while not done:
                    # img, depth = self.env.sim.render(mode='offscreen', camera_name='track', width=256, height=256, depth=True)
                    img, depth = self.env.sim.render(mode='offscreen', width=256, height=256, depth=True)
                    act,pred_obs = self.policy.select_action_from_img(img,imgmodel,obs[2:4])
                    # pred_obs = obs
                    gt.append(obs)
                    pred.append(pred_obs)
                    obs, r, done, info = self.env.step(act)
                    episode_r += r
                reward_buffer.append(episode_r)
                l1_loss = abs(np.array(pred)-np.array(gt)).mean()
                print(episode_r,l1_loss)

            episode_r = sum(reward_buffer)
            print('average reward: {}'.format(episode_r/episode_n))
            return np.array(reward_buffer)

