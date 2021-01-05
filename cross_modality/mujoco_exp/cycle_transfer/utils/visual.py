

import argparse
from scipy.stats import laplace
import scipy.stats
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os
from torch.distributions.laplace import Laplace
import torch


def pdf(arr1,num_bins=100,left=-3,right=3):
    bins = np.linspace(left, right, num=num_bins)
    PDF1 = pd.cut(arr1,bins).value_counts() / len(arr1)
    return PDF1/((right-left+0.0)/num_bins)


class Count:
    def __init__(self,opt):
        self.opt = opt
        self.log_root = opt.log_root
        self.env_logs = os.path.join(self.log_root, '{}_data'.format(self.opt.env))
        self.data_root = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type, self.opt.data_id))
        self.state = np.load(os.path.join(self.data_root,'now_state.npy'))
        self.draw_dpf(self.state)

    def draw_dpf(self,data):
        num_bins = 100
        ncols = int(np.sqrt(data.shape[1]))+1
        nrows = int(np.sqrt(data.shape[1]))+1
        assert (ncols*nrows>=data.shape[1])
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i>=data.shape[1]:
                continue
            e = data[:, ax_i]

            left = min(e.min(), -e.max())
            right = max(e.max(), -e.min())

            param = laplace.fit(e)
            print(ax_i, param)

            try:
                tmp = pdf(e, num_bins, left, right)
                index = tmp.index.T._data
                values = tmp.values
            except:
                print(ax_i,'error')
            ax.plot([x.left for x in index], values)
        plt.savefig(os.path.join(self.data_root,'statepdf.jpg'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control dataset analyzer')
    # parser.add_argument("--env", default="Ant-v2")
    # parser.add_argument("--env", default="Pendulum-v0")
    # parser.add_argument("--env", default="InvertedDoublePendulum-v2")
    # parser.add_argument("--env", default="Reacher-v2")
    # parser.add_argument("--env", default="Swimmer-v2")
    parser.add_argument("--env", default="Walker2d-v2")
    parser.add_argument("--log_root", default="../../../../../../cross_modality")
    parser.add_argument('--data_type', type=str, default='base', help='data type')
    parser.add_argument('--data_id', type=int, default=0, help='data id')
    opt = parser.parse_args()

    agent = Count(opt)

