
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from options import get_options
from cycle.dyncycle import CycleGANModel
from cycle.data import CycleData
from cycle.utils import init_logs

def main(args):
    txt_logs, img_logs, weight_logs = init_logs(args)
    model = CycleGANModel(args)
    model.load(weight_logs)
    model.cross_policy.eval_policy(
                        gxmodel=gxmodel,
                        axmodel=axmodel,
                        eval_episodes=10)
    # model.cross_policy.eval_policy(
    #                     gxmodel=model.netG_B,
    #                     axmodel=model.net_action_G_A,
    #                     eval_episodes=10)


def analysis():
    data_agent = CycleData(args)
    now1 = np.load(os.path.join(data_agent.data_root1,'now_state.npy'))
    act1 = np.load(os.path.join(data_agent.data_root1,'action.npy'))
    now2 = np.load(os.path.join(data_agent.data_root2,'now_state.npy'))
    act2 = np.load(os.path.join(data_agent.data_root2,'action.npy'))
    # print(now1.std(0))
    # print(now2.std(0))
    print(act1.std(0))
    print(act2.std(0))


def gxmodel(state):
    state = state.squeeze().cpu().data.numpy()
    # new_state = list(state[:2])
    # new_state.extend(list(state[3:8]))
    # new_state.append(state[9])
    new_state = [state[0],state[1],state[2],
                 state[4],state[5],state[6],state[7],state[8]]
    return torch.tensor(np.array(new_state)).float().cuda()


def axmodel(action):
    action = action.squeeze().cpu().data.numpy()
    p = (random.random() - 0.5) * 2
    new_action = list(action)
    new_action.append(p)
    # new_action = [p]
    # new_action.extend(list(action))
    return torch.tensor(np.array(new_action)).float().cuda()


if __name__ == '__main__':
    args = get_options()
    # args.env = 'Swimmer-v2'
    args.env = 'HalfCheetah-v2'
    args.pretrain_f = True
    args.data_type1 = 'base'
    args.data_type2 = '3leg'
    args.data_id1 = 0
    args.data_id2 = 0
    args.pair_n = 5000
    args.state_dim1 = 8
    args.action_dim1 = 2
    args.state_dim2 = 10
    args.action_dim2 = 3

    analysis()
    # main(args)
