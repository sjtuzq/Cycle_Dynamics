from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict

import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable


def init_logs(opt):
    log_dir = './explogs{}'.format(opt.exp_id)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if opt.istrain:
        img_logs = os.path.join(log_dir, 'train')
    else:
        img_logs = os.path.join(log_dir,'eval')
    weight_logs = os.path.join(log_dir, 'weights')
    if not os.path.exists(img_logs):
        os.mkdir(img_logs)
    if not os.path.exists(weight_logs):
        os.mkdir(weight_logs)
    return img_logs,weight_logs

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

class DSPdata():
    def __init__(self):
        try:
            # is main file is data.py
            file = 'dsprites.npz'
            self.dataset_zip = np.load(file,encoding='bytes')
        except:
            # if main file is toyexp.py
            file = './expmodule/dsprites.npz'
            self.dataset_zip = np.load(file,encoding='bytes')
        self.imgs = self.dataset_zip['imgs']
        self.latents_values = self.dataset_zip['latents_values']
        self.latents_classes = self.dataset_zip['latents_classes']
        self.metadata = self.dataset_zip['metadata'][()]
        # Define number of values per latents and functions to convert to indices
        self.latents_sizes = self.metadata[b'latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],np.array([1,])))

        """
        latent meaning:
        color, shape, scale, orientation, posX, posY
        """


    def latent_to_index(self,latents):
        return np.dot(latents, self.latents_bases).astype(int)


    def sample_latent(self,size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples


    # Helper function to show images
    def show_images_grid(self,imgs_, num_images=8):
        # num_images = min(imgs.shape[0],num_images)
        ncols = int(np.ceil(num_images ** 0.5))
        nrows = int(np.ceil(num_images / ncols))
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i < num_images:
                ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')


    def show_density(self,imgs):
        _, ax = plt.subplots()
        ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
        ax.grid('off')
        ax.set_xticks([])
        ax.set_yticks([])

    def sample(self,size=5000,flag='random'):
        if flag == 'random':
            return self.random_sample(size=size)
        elif flag == 'vary_xy':
            return self.vary_xy_sample(size=size)

    def random_sample(self,size=5000):
        # Sample latents randomly
        latents_sampled = self.sample_latent(size=size)

        # Select images
        indices_sampled = self.latent_to_index(latents_sampled)
        imgs_sampled = self.imgs[indices_sampled]

        # Show images
        # show_images_grid(imgs_sampled)

        # Compute the density of the data to show that no pixel ever goes out of
        # the boundary. Obviously it also means that the main support of the pixels is in the center
        # half.
        # Locations cover a square, which make the aligned X-Y latents more likely for
        # models to discover.
        # show_density(imgs_sampled)
        return imgs_sampled


    def fix_posx_sample(self,size=5000):
        ## Fix posX latent to left
        latents_sampled = self.sample_latent(size=5000)
        latents_sampled[:, -2] = 0
        indices_sampled = self.latent_to_index(latents_sampled)
        imgs_sampled = self.imgs[indices_sampled]
        return imgs_sampled

    def fix_ori_sample(self,size=5000):
        ## Fix orientation to 0.8 rad
        latents_sampled = self.sample_latent(size=5000)
        latents_sampled[:, 3] = 5
        indices_sampled = self.latent_to_index(latents_sampled)
        imgs_sampled = self.imgs[indices_sampled]
        return imgs_sampled

    def vary_xy_sample(self,size=5000):
        latents_sampled = self.sample_latent(size=size)
        latents_sampled[:, 1] = 1
        latents_sampled[:, 2] = 1
        latents_sampled[:, 3] = 5
        indices_sampled = self.latent_to_index(latents_sampled)
        imgs_sampled = self.imgs[indices_sampled]
        return imgs_sampled

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
