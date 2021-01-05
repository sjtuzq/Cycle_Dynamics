

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

class Visual:
    def __init__(self,args):
        self.opt = args
        self.env_logs = os.path.join(self.opt.log_root, '{}_data'.format(self.opt.env))
        self.data_folder = os.path.join(self.env_logs,
                    '{}_{}'.format(self.opt.data_type,self.opt.data_id))

        origin_path = os.path.join(self.opt.log_root, '{}_base/models'.format(self.opt.env))
        self.origin = np.load(os.path.join(origin_path, 'origin.npy'))
        self.ours = np.load(os.path.join(self.data_folder,'ours.npy'))
        self.base = np.load(os.path.join(self.data_folder,'baseline.npy'))

        self.exp_n = self.ours.shape[0]

    def cmp(self):
        plt.plot(range(self.exp_n),self.ours.mean(1),color='r',label='mean_ours')
        # plt.plot(range(self.exp_n),self.base.mean(1),color='b',label='mean_base')
        plt.plot(range(self.exp_n),[self.base.mean()]*self.exp_n,color='b',label='mean_base')

        plt.plot(range(self.exp_n), self.ours.std(1), color='g',label='std_ours')
        # plt.plot(range(self.exp_n), self.base.std(1), color='y',label='std_base')
        plt.plot(range(self.exp_n), [self.base.std()]*self.exp_n, color='y',label='std_base')
        plt.legend()
        plt.savefig(os.path.join(self.data_folder,'cmp.jpg'))

        ours = self.ours.mean(1)[-1]
        base = self.base.mean()
        print('finished compare!  ours:{}  baseline:{}'.format(ours,base))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control dataset analyzer')
    parser.add_argument("--env", default="HalfCheetah-v2")
    # parser.add_argument("--env", default="Hopper-v2")
    parser.add_argument("--force", type=bool, default=False)
    parser.add_argument("--log_root", default="../../../../cross_physics_snapshot")
    # parser.add_argument("--log_root", default="../../../../cross_physics")
    parser.add_argument('--data_type', type=str, default='arma2', help='data type')
    parser.add_argument('--data_id', type=int, default=0, help='data id')
    parser.add_argument('--episode_n', type=int, default=100, help='episode number')
    opt = parser.parse_args()

    agent = Visual(opt)
    agent.cmp()