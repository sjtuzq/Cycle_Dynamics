

import os
import numpy as np
import matplotlib.pyplot as plt

data_root = '../../../../../reallogs/explog5'
now = np.load(os.path.join(data_root,'now.npy'))
pred = np.load(os.path.join(data_root,'pred.npy'))
gt = np.load(os.path.join(data_root,'gt.npy'))

delta = pred-gt
plt.scatter(delta[:,0],delta[:,1],s=5)
plt.savefig(os.path.join(data_root,'delta2.jpg'))

# plt.scatter(now[:100,2],now[:100,3])
# plt.savefig(os.path.join(data_root,'now.jpg'))

