
import os
import gym
import torch
import random
import argparse
import dmc2gym
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
        env_logs = os.path.join(opt.log_root, '{}_{}_data'.format(opt.domain_name, opt.task_name))
        data_path = os.path.join(env_logs, '{}_{}'.format(opt.data_type1, opt.data_id1))
        self.mean_std_path = os.path.join(data_path,'now_state.npy')
        self.data = np.load(self.mean_std_path)
        self.mean = torch.tensor(self.data.mean(0)).float().cuda()
        self.std = torch.tensor(self.data.std(0)).float().cuda()
        print(self.mean,self.std)
        self.clip_range = 5

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def select_action_from_img(self,img,imgmodel):
        img = self.stack_agent.push(img)
        output = imgmodel(img.unsqueeze(0)).squeeze()

        mask = torch.zeros(self.mean.shape[0]-output.shape[0]).cuda()
        pred_state = torch.cat((output,mask),0)
        pred_state = pred_state * self.std + self.mean

        # pred_state[2] = 0
        # pred_state[3] = 0
        # pred_state[6:] = 0
        # pred_state[:2] = pred_state[:2].clamp(max=2.5,min=-2.5)

        pred_state = pred_state.cpu().data.numpy()
        action = self.select_action(pred_state)
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
        self.env = dmc2gym.make(
            domain_name=opt.domain_name,
            task_name=opt.task_name,
            seed=0,
            visualize_reward=False,
            from_pixels=True,
            height=256,
            width=256,
            frame_skip=opt.frame_skip
        )

        self.env.seed(0)
        random.seed(0)
        # self.state_dim = self.env.observation_space.shape[0]
        if self.opt.domain_name=='finger':
            self.state_dim = 9
        elif self.opt.domain_name=='reacher':
            self.state_dim = 6
        # self.state_dim = self.env.observation_space.shape[0] if opt.state_dim==0 else opt.state_dim

        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.log_root = opt.log_root
        self.episode_n = opt.episode_n
        log_path = os.path.join(opt.log_root, '{}_{}_base'.format(opt.domain_name, opt.task_name))
        self.policy_path = os.path.join(log_path, 'models/TD3_{}_0_actor'.format(opt.domain_name))
        if self.opt.domain_name=='finger':
            self.policy_path = os.path.join(log_path, 'models_bak/TD3_{}_0_actor'.format(opt.domain_name))
        self.policy = TD3(self.policy_path,self.state_dim,self.action_dim,self.max_action,opt)

    def online_test(self,imgmodel,episode_n=100,imgpath=None):
        with torch.no_grad():
            gt,pred = [],[]
            reward_buffer = []

            save_flag = False
            if imgpath is not None:
                if not os.path.exists(imgpath):
                    os.mkdir(imgpath)
                save_flag = True

            for episode in tqdm(range(episode_n)):
                obs = self.env.reset()
                done = False
                episode_r = 0.
                if save_flag:
                    episode_path = os.path.join(imgpath, 'episode_{}'.format(episode))
                    if not os.path.exists(episode_path):
                        os.mkdir(episode_path)
                count = 0
                while not done:
                    img = obs.transpose(1,2,0)
                    state = self.env.current_state
                    act,pred_obs = self.policy.select_action_from_img(img,imgmodel)
                    # act = self.policy.select_action(self.env.current_state)
                    # act = self.env.action_space.sample()

                    pred.append(pred_obs)
                    gt.append(self.env.current_state)
                    obs, r, done, info = self.env.step(act)
                    episode_r += r
                    if save_flag:
                        Image.fromarray(img).save(os.path.join(episode_path, 'img_{}.jpg'.format(count)))
                    count += 1

                reward_buffer.append(episode_r)
                l1_loss = abs(np.array(pred)-np.array(gt)).mean(0)
                print(episode_r,l1_loss)
                # import pdb
                # pdb.set_trace()
                # print(episode_r)

            episode_r = sum(reward_buffer)
            np.save('./pred.npy',np.array(pred))
            np.save('./gt.npy',np.array(gt))
            print('average reward: {}'.format(episode_r/episode_n))
            return np.array(reward_buffer)
