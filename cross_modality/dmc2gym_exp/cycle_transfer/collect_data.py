
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

from utils.visual import Count


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
    def __init__(self,policy_path,state_dim,action_dim,max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.weight_path = policy_path
        self.actor.load_state_dict(torch.load(self.weight_path))
        print('policy weight loaded!')

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
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.log_root = opt.log_root
        self.episode_n = opt.episode_n
        # self.policy_path = os.path.join(opt.log_root,
        #                 '{}_base/models/TD3_{}_0_actor'.format(opt.env,opt.env))
        # self.policy = TD3(self.policy_path,self.state_dim,self.action_dim,self.max_action)
        self.setup(opt)
        print('start collect')
        self.create_data()
        self.count = Count(opt)
        print('----------- Dataset initialized ---------------')
        print('-----------------------------------------------\n')

    def setup(self,opt):
        self.episode_n = opt.episode_n
        self.env_logs = safe_path(os.path.join(self.log_root, '{}_{}_data'.format(self.opt.domain_name,self.opt.task_name)))
        self.data_root = safe_path(os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type, self.opt.data_id)))
        self.img_path = safe_path(os.path.join(self.data_root,'imgs'))

    def create_data(self):
        self.reset_buffer()
        for i_episode in range(self.episode_n):
            print(i_episode)
            observation, done, t = self.env.reset(), False, 0

            episode_path = os.path.join(self.img_path,'episode-{}'.format(i_episode))
            if not os.path.exists(episode_path):
                os.mkdir(episode_path)

            while not done:
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                # print(observation.shape)
                self.add_action(action)
                state = self.env.current_state
                self.add_observation(state)

                path = os.path.join(episode_path, 'img_{}_{}.jpg'.format(i_episode, t + 1))
                self.check_and_save(observation,path)
                t += 1

                if done:
                    print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                    break
            self.merge_buffer()

        self.collect_data()

    def check_and_save(self,img,path):
        img = Image.fromarray(img.transpose(1,2,0))
        img.save(path)

    def collect_data(self):
        self.env.close()
        self.norm_state()
        self.pair_n = self.now_state.shape[0]
        assert (self.pair_n == self.next_state.shape[0])
        assert (self.pair_n == self.action.shape[0])
        self.save_npy()

    def norm_state(self):
        self.now_state = np.vstack(self.now_state)
        self.next_state = np.vstack(self.next_state)
        self.action = np.vstack(self.action)

    def save_npy(self):
        np.save(os.path.join(self.data_root,'now_state.npy'),self.now_state)
        np.save(os.path.join(self.data_root,'next_state.npy'),self.next_state)
        np.save(os.path.join(self.data_root,'action.npy'),self.action)

    def reset_buffer(self):
        self.joint_pose_buffer = []
        self.achieved_goal_buffer = []
        self.goal_pos_buffer = []
        self.action_buffer = []

        self.now_state = []
        self.next_state = []
        self.action = []

    def add_observation(self,observation):
        self.joint_pose_buffer.append(observation)

    def add_action(self,action):
        self.action_buffer.append(action)

    def merge_buffer(self):
        self.now_state += self.joint_pose_buffer[:-1]
        self.next_state += self.joint_pose_buffer[1:]
        self.action += self.action_buffer[1:]

        self.joint_pose_buffer = []
        self.achieved_goal_buffer = []
        self.goal_pos_buffer = []
        self.action_buffer = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control dataset analyzer')
    # parser.add_argument("--env", default="Pendulum-v0")
    parser.add_argument("--domain_name", default="reacher")
    parser.add_argument("--task_name", default="hard")
    parser.add_argument("--force", type=bool, default=False)
    parser.add_argument("--log_root", default="../../../../../cross_modality")
    parser.add_argument('--data_type', type=str, default='base', help='data type')
    parser.add_argument('--data_id', type=int, default=3, help='data id')
    parser.add_argument('--frame_skip', type=int, default=2, help='data id')
    parser.add_argument('--episode_n', type=int, default=20, help='episode number')
    opt = parser.parse_args()

    dataset = CycleData(opt)

