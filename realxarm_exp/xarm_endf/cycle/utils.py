from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict

import os
import random
import torch
import shutil
from glob import glob
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def init_logs(opt):
    log_dir = safe_path(os.path.join(opt.data_root,'explog{}'.format(opt.exp_id)))
    if opt.istrain:
        img_logs = safe_path(os.path.join(log_dir, 'train'))
    else:
        img_logs = safe_path(os.path.join(log_dir,'eval'))
    weight_logs = safe_path(os.path.join(log_dir, 'weights'))
    script_logs = safe_path(os.path.join(log_dir,'scripts'))

    if opt.istrain:
        #backup scripts
        for dir in os.listdir('./'):
            if os.path.isdir(os.path.join('./',dir)):
                safe_path(os.path.join(script_logs,dir))
        for fold in ['data','model','visual']:
            safe_path(os.path.join(script_logs, 'cycle/{}'.format(fold)))
        for file in (glob('*.py')+glob('*/*.py')+glob('*/*/*.py')):
            shutil.copy(file,os.path.join(script_logs,file))

    tensor_writer = SummaryWriter(safe_path(os.path.join(log_dir,'tensorlogs')))
    return img_logs,weight_logs,tensor_writer

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def my_load(model,pretrained_dict):
    current_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for key in current_dict.keys():
        if key in pretrained_dict.keys():
            new_state_dict[key] = pretrained_dict[key]
        elif 'encoder1' in key:
            if pretrained_dict[key.replace('encoder1','encoder')].shape == current_dict[key].shape:
                new_state_dict[key] = pretrained_dict[key.replace('encoder1','encoder')]
            else:
                print("weight {} lost!".format(key))
                new_state_dict[key] = current_dict[key]
        elif 'encoder2' in key:
            if pretrained_dict[key.replace('encoder2', 'encoder')].shape == current_dict[key].shape:
                new_state_dict[key] = pretrained_dict[key.replace('encoder2','encoder')]
            else:
                print("weight {} lost!".format(key))
                new_state_dict[key] = current_dict[key]
        else:
            print('weight {} lost!'.format(key))
            new_state_dict[key] = current_dict[key]
    model.load_state_dict(new_state_dict)
    return model


import imageio
import os

def compose_gif(folder,gif_file):
    img_paths = os.listdir(folder)
    gif_images = []
    for i,path in enumerate(img_paths[:30]):
        if i%2==0:
            continue
        path = os.path.join(folder,path)
        gif_images.append(imageio.imread(path))
    imageio.mimsave(gif_file,gif_images,fps=3)

def main():
    # agent = DSPdata()
    # data = agent.sample(1000,'vary_xy')
    # print(data.shape)
    for exp_id in range(1,6):
        folder = '../explogs{}/eval'.format(exp_id)
        gif_file = '../eval_{}.gif'.format(exp_id)
        compose_gif(folder,gif_file)

if __name__ == '__main__':
    main()
