
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from options import get_options
from cycle.data import CycleData
from cycle.dyncycle import CycleGANModel
from cycle.utils import init_logs


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


def add_errors(model,display):
    errors = model.get_current_errors()
    for key, value in errors.items():
        if key=='G_act_B':
            display += '\n'
        display += '{}:{:.4f}  '.format(key, value)
    return display


def train(args):
    txt_logs, img_logs, weight_logs = init_logs(args)
    data_agent = CycleData(args)
    model = CycleGANModel(args)
    model.fengine.train_statef(data_agent.data1)
    model.cross_policy.eval_policy(
                        gxmodel=model.netG_B,
                        axmodel=model.net_action_G_A,
                        eval_episodes=10)

    best_reward = 0
    end_id = 0
    for iteration in range(3):

        args.lr_Gx = 1e-4
        args.lr_Ax = 0
        model.update(args)

        start_id = end_id
        end_id = start_id + args.pair_n
        for batch_id in range(start_id,end_id):
            item = data_agent.sample()
            data1,data2 = item
            model.set_input(item)
            model.optimize_parameters()
            real,fake = model.fetch()

            if (batch_id + 1) % args.display_gap == 0:
                display = '\n===> Batch[{}/{}]'.format(batch_id+1, args.pair_n)
                print(display)
                display = add_errors(model, display)
                txt_logs.write('{}\n'.format(display))
                txt_logs.flush()

                path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id + 1))
                model.visual(path)

            if (batch_id + 1) % args.eval_gap == 0:
                reward = model.cross_policy.eval_policy(
                    gxmodel=model.netG_B,
                    axmodel=model.net_action_G_A,
                    eval_episodes=args.eval_n)
                if reward>best_reward:
                    best_reward = reward
                    model.save(weight_logs)
                print('best_reward:{:.1f}  cur_reward:{:.1f}'.format(best_reward,reward))

        args.init_start = False
        args.lr_Gx = 0
        args.lr_Ax = 1e-4
        model.update(args)

        start_id = end_id
        end_id = start_id + args.pair_n
        for batch_id in range(start_id,end_id):
            item = data_agent.sample()
            data1, data2 = item
            model.set_input(item)
            model.optimize_parameters()
            real, fake = model.fetch()

            if (batch_id + 1) % args.display_gap == 0:
                display = '\n===> Batch[{}/{}]'.format(batch_id+1, args.pair_n)
                print(display)
                display = add_errors(model, display)
                txt_logs.write('{}\n'.format(display))
                txt_logs.flush()

                path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id + 1))
                model.visual(path)

            if (batch_id + 1) % args.eval_gap == 0:
                reward = model.cross_policy.eval_policy(
                    gxmodel=model.netG_B,
                    axmodel=model.net_action_G_A,
                    eval_episodes=args.eval_n)
                if reward>best_reward:
                    best_reward = reward
                    model.save(weight_logs)

                print('best_reward:{:.1f}  cur_reward:{:.1f}'.format(best_reward,reward))


def test(args):
    args.istrain = False
    args.init_start = False
    txt_logs, img_logs, weight_logs = init_logs(args)
    data_agent = CycleData(args)
    model = CycleGANModel(args)
    model.fengine.train_statef(data_agent.data1)
    print(weight_logs)
    model.load(weight_logs)
    model.update(args)

    model.cross_policy.eval_policy(
        gxmodel=model.netG_B,
        axmodel=model.net_action_G_A,
        # imgpath=img_logs,
        eval_episodes=10)



if __name__ == '__main__':
    args = get_options()

    # args.env = 'HalfCheetah-v2'
    # args.pretrain_f = False
    # args.data_type1 = 'base'
    # args.data_type2 = '3leg'
    # args.data_id1 = 1
    # args.data_id2 = 1
    # args.pair_n = 7000
    # args.eval_gap = 1000
    # args.state_dim1 = 17
    # args.action_dim1 = 6
    # args.state_dim2 = 23
    # args.action_dim2 = 9

    train(args)
    args.istrain = False
    with torch.no_grad():
        test(args)


    # args.env = 'Swimmer-v2'
    # args.pretrain_f = True
    # args.data_type1 = 'base'
    # args.data_type2 = '4part'
    # args.data_id1 = 0
    # args.data_id2 = 0
    # args.pair_n = 5000
    # args.eval_gap = 1000
    # args.state_dim1 = 8
    # args.action_dim1 = 2
    # args.state_dim2 = 10
    # args.action_dim2 = 3
    # #
    # # train(args)
    # with torch.no_grad():
    #     test(args)


