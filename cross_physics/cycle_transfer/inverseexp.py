

import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from collect_data import CycleData
from models import Inversemodel,Axmodel,net_init


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


def show_points(gt_data, pred_data):
    ncols = int(np.sqrt(gt_data.shape[1])) + 1
    nrows = int(np.sqrt(gt_data.shape[1])) + 1
    assert (ncols * nrows >= gt_data.shape[1])
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i >= gt_data.shape[1]:
            continue
        ax.scatter(gt_data[:, ax_i], pred_data[:, ax_i], s=3, label='xyz_{}'.format(ax_i))

class Agent():
    def __init__(self,opt):
        self.opt = opt
        self.dataset = CycleData(opt)
        self.model = self.dataset.inverse_model

class ActionAgent:
    def __init__(self,opt):
        self.opt = opt
        opt.data_type = opt.data_type1
        opt.data_id = opt.data_id1
        self.agent1 = Agent(opt)
        opt.data_type = opt.data_type2
        opt.data_id = opt.data_id2
        self.agent2 = Agent(opt)

        opt.state_dim = self.agent1.dataset.state_dim
        opt.action_dim = self.agent1.dataset.action_dim
        self.env_logs = self.agent1.dataset.env_logs

        self.model = Axmodel(opt).cuda()
        # net_init(self.model)
        self.weight_path = os.path.join(self.env_logs,'inv_model_{}_{}.pth'
                                    .format(opt.data_type1,opt.data_type2))

    def train_ax(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adam(self.model_1_2.parameters(), lr=3e-4)
        loss_fn = nn.L1Loss()

        print('----------initial test as baseline--------------')
        ref_reward = self.agent2.dataset.online_test(lambda x,y:y,10)

        minimum_loss = 0.
        max_reward = 0
        self.opt.istrain = True
        for epoch in range(30):
            epoch_loss, cmp_loss = 0, 0
            if epoch == 10:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
            elif epoch == 20:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            self.agent1.dataset.reset()
            for i in (range(self.opt.pair_n)):
                item = self.agent1.dataset.sample()
                now_state, action, nxt_state = item
                # pred1 = self.agent1.model(now_state, nxt_state).detach()
                pred2 = self.agent2.model(now_state, nxt_state).detach()

                out = self.model(now_state, action)
                loss = loss_fn(out, pred2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print('epoch:{} loss:{:.4f}'.format(epoch, epoch_loss / self.opt.pair_n))

            reward_ours = self.agent2.dataset.online_test(self.model,5)

            if reward_ours.mean() > max_reward:
                max_reward = reward_ours.mean()
                torch.save(self.model.state_dict(), self.weight_path)
            print('ours_cur:{:.2f}  ours_max:{:.2f}  ref_baseline:{:.2f}'
                  .format(reward_ours.mean(),max_reward,ref_reward.mean()))

        self.opt.istrain = False
        self.dataset = CycleData(self.opt)
        epoch_loss, cmp_loss = 0, 0
        for i in (range(self.opt.pair_n)):
            item = self.agent1.dataset.sample()
            now_state, action, nxt_state = item
            # pred1 = self.agent1.model(now_state, nxt_state).detach()
            pred2 = self.agent2.model(now_state, nxt_state).detach()
            out = self.model(now_state, action)
            loss = loss_fn(out, pred2)
            epoch_loss += loss.item()
        print('evaluation loss:{:.2f}'.format(epoch_loss / self.opt.pair_n))
        self.opt.istrain = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control dataset analyzer')
    parser.add_argument('--istrain', type=bool, default=True, help='train or eval')
    parser.add_argument('--pair_n', type=int, default=3000, help='dataset sample number')

    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--force", type=bool, default=False)
    parser.add_argument("--log_root", default="../../../logs/cross_physics")
    parser.add_argument('--episode_n', type=int, default=1, help='episode number')
    parser.add_argument('--state_dim', type=int, default=0, help='state dim')
    parser.add_argument('--action_dim', type=int, default=0, help='action dim')

    parser.add_argument('--data_type', type=str, default='base', help='data type')
    parser.add_argument('--data_type1', type=str, default='base', help='data type')
    parser.add_argument('--data_type2', type=str, default='arma3', help='data type')

    parser.add_argument('--data_id', type=int, default=0, help='data id')
    parser.add_argument('--data_id1', type=int, default=1, help='data id')
    parser.add_argument('--data_id2', type=int, default=1, help='data id')

    opt = parser.parse_args()

    agent = ActionAgent(opt)
    agent.train_ax()

