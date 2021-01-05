

import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.utils.data as Data
import matplotlib.pyplot as plt

from collect_data import CycleData
from models import Forwardmodel,Axmodel,Dmodel,GANLoss,ImagePool,net_init


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
        self.model = self.dataset.model


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
        self.back_model = Axmodel(opt).cuda()
        self.dmodel = Dmodel(opt).cuda()
        if self.opt.env == 'Walker2d-v2':
            net_init(self.model)
            net_init(self.back_model)
            net_init(self.dmodel)

        self.criterionGAN = GANLoss().cuda()
        self.fake_pool = ImagePool(256)
        self.real_pool = ImagePool(256)
        self.weight_path = os.path.join(self.env_logs,
                'model_{}_{}.pth'.format(opt.data_type1,opt.data_type2))

    def get_optim(self,lr):
        optimizer_g = torch.optim.Adam([{'params': self.agent1.model.parameters(), 'lr': 0.0},
                                      {'params': self.back_model.parameters(), 'lr': lr},
                                      {'params': self.model.parameters(), 'lr': lr}])
        self.optimizer_d = torch.optim.Adam(self.dmodel.parameters(),lr=lr)
        return optimizer_g,self.optimizer_d

    def cal_gan(self,real_action,trans_action):
        ######################
        # (1) Update D network
        ######################

        self.optimizer_d.zero_grad()

        fake_b = trans_action
        fake_ab = self.fake_pool.query(fake_b.detach())
        pred_fake = self.dmodel(fake_ab)
        loss_d_fake = self.criterionGAN(pred_fake, False)

        real_b = real_action
        real_ab = self.real_pool.query(real_b)
        pred_real = self.dmodel(real_ab)
        loss_d_real = self.criterionGAN(pred_real, True)

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward()
        self.optimizer_d.step()

        ######################
        # (2) Update G network
        ######################
        fake_ab = fake_b
        pred_fake = self.dmodel(fake_ab)
        loss_g_gan = self.criterionGAN(pred_fake, True)

        return loss_g_gan

    def train_ax(self):
        lr = 1e-3
        optimizer_g,optimizer_d = self.get_optim(lr)
        loss_fn = nn.L1Loss()

        print('----------initial test as baseline--------------')
        ref_reward = self.agent2.dataset.online_test(lambda x,y:y,10)

        ours,baseline = [],[]
        self.opt.istrain = True
        last_back_loss = 100.
        max_reward = 0.
        for epoch in range(self.opt.epoch_n):
            epoch_loss, cmp_loss = 0, 0
            if self.opt.env == 'HalfCheetah-v2':
                if epoch == 10:
                    lr = 3e-4
                    optimizer_g,optimizer_d = self.get_optim(lr)
                elif epoch == 20:
                    lr = 1e-4
                    optimizer_g,optimizer_d = self.get_optim(lr)

            for i in (range(self.opt.pair_n)):
                item1 = self.agent1.dataset.sample()
                real_action = item1[1]

                item2 = self.agent2.dataset.sample()
                now_state, action, nxt_state = item2

                trans_action = self.model(now_state,action)
                out = self.agent1.model(now_state, trans_action)
                loss_cycle = loss_fn(out, nxt_state)*20
                loss = loss_cycle

                back_action = self.back_model(now_state,trans_action)
                loss_back = loss_fn(back_action,action)
                loss += loss_back

                loss_g_gan = self.cal_gan(real_action,trans_action)
                loss_all = loss + loss_g_gan
                # loss_all = loss

                if self.opt.istrain:
                    optimizer_g.zero_grad()
                    loss_all.backward()
                    optimizer_g.step()
                epoch_loss += loss_cycle.item()
                cmp_loss += loss_back.item()
            print('epoch:{} cycle_loss:{:.3f}  back_loss:{:.3f}'
                  .format(epoch, epoch_loss / self.opt.pair_n, cmp_loss / self.opt.pair_n))

            reward_ours = self.agent2.dataset.online_test(self.back_model,self.opt.eval_n)

            if reward_ours.mean() > max_reward:
                max_reward = reward_ours.mean()
                torch.save(self.back_model.state_dict(), self.weight_path)
            print('ours_cur:{:.2f}  ours_max:{:.2f}  ref_baseline:{:.2f}\n'
                  .format(reward_ours.mean(), max_reward, ref_reward.mean()))

    def eval_ax(self):
        self.back_model.load_state_dict(torch.load(self.weight_path))
        img_path_ours = os.path.join(self.agent2.dataset.data_folder,'ours_eval')
        img_path_base = os.path.join(self.agent2.dataset.data_folder,'base_eval')
        reward_ours = self.agent2.dataset.online_test(self.back_model, 10,img_path_ours)
        reward_base = self.agent2.dataset.online_test(lambda x, y: y, 10,img_path_base)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control dataset analyzer')
    parser.add_argument('--istrain', type=bool, default=False, help='train or eval')
    parser.add_argument('--pretrainF', type=bool, default=False, help='train or eval')
    parser.add_argument('--pair_n', type=int, default=3000, help='dataset sample number')

    parser.add_argument("--env", default="Walker2d-v2")
    parser.add_argument("--force", type=bool, default=False)
    parser.add_argument("--log_root", default="../../../logs/cross_physics")
    parser.add_argument('--episode_n', type=int, default=100, help='episode number')
    parser.add_argument('--state_dim', type=int, default=0, help='state dim')
    parser.add_argument('--action_dim', type=int, default=0, help='action dim')
    parser.add_argument('--eval_n', type=int, default=100, help='evaluation episode number')
    parser.add_argument('--epoch_n', type=int, default=30, help='training epoch number')

    parser.add_argument('--data_type', type=str, default='base', help='data type')
    parser.add_argument('--data_type1', type=str, default='base', help='data type')
    parser.add_argument('--data_type2', type=str, default='arma3', help='data type')

    parser.add_argument('--data_id', type=int, default=0, help='data id')
    parser.add_argument('--data_id1', type=int, default=1, help='data id')
    parser.add_argument('--data_id2', type=int, default=1, help='data id')

    opt = parser.parse_args()

    opt.env = 'HalfCheetah-v2'
    opt.eval_n = 5
    opt.pair_n = 3000
    opt.istrain = True
    opt.epoch_n = 30
    # opt.data_id1 = 4
    # opt.data_id2 = 4
    opt.data_type2 = 'arma3'
    agent = ActionAgent(opt)
    agent.train_ax()

    # opt.env = "Swimmer-v2"
    # opt.eval_n = 10
    # opt.pair_n = 3000
    # opt.istrain = False
    # opt.data_type2 = 'density1400'
    # agent = ActionAgent(opt)
    # agent.train_ax()

    # opt.env = "Walker2d-v2"
    # opt.eval_n = 100
    # opt.pair_n = 3000
    # opt.data_type2 = 'mass2'
    # agent = ActionAgent(opt)
    # # agent.train_ax()
    # agent.eval_ax()

    # opt.env = 'Hopper-v2'
    # opt.eval_n = 100
    # opt.pair_n = 500
    # opt.data_type2 = 'mass2'
    # agent = ActionAgent(opt)
    # # agent.train_ax()
    # agent.eval_ax()



