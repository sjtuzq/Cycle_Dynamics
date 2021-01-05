

import numpy as np
import matplotlib.pyplot as plt

def cmp(data,gt):
    ncols = int(np.sqrt(data.shape[1])) + 1
    nrows = int(np.sqrt(data.shape[1])) + 1
    assert (ncols * nrows >= data.shape[1])
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        try:
            ax.scatter(data[:,ax_i],gt[:,ax_i])
            print(ax_i)
        except:
            continue
    plt.savefig('../analysis.jpg')



if __name__ == '__main__':
    pred = np.load('../pred.npy')
    gt = np.load('../gt.npy')
    cmp(pred,gt)
