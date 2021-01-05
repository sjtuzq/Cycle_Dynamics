

import os
import gym
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools
import dmc2gym

from model.dgmodel import state2img,imgDmodel,stateDmodel
from model.encoder4 import PixelEncoder as img2state
from utils.utils import ImagePool,GANLoss
from model.fmodel import Fmodel,ImgFmodel,ADmodel,AGmodel
from model.imgpolicy import ImgPolicy

class CycleGANModel():
    def __init__(self,opt):
        self.opt = opt
        self.isTrain = opt.istrain
        self.env = dmc2gym.make(
            domain_name=opt.domain_name,
            task_name=opt.task_name,
            seed=0,
            visualize_reward=False,
            from_pixels=True,
            height=256,
            width=256,
            frame_skip=opt.frame_skip
        )

        self.env.seed(0)
        # self.state_dim = self.env.observation_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0] if opt.state_dim==0 else opt.state_dim
        self.action_dim = self.env.action_space.shape[0]
        if self.opt.action_dim == 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.opt.action_dim

        opt.state_dim = self.state_dim
        opt.action_dim = self.action_dim
        self.max_action = float(self.env.action_space.high[0])
        self.img_policy = ImgPolicy(opt)

        self.Tensor = torch.cuda.FloatTensor
        self.netG_A = img2state(opt=self.opt).cuda()
        self.netG_B = img2state(opt=self.opt).cuda()
        self.net_action_G_A = AGmodel(flag='A2B',opt=self.opt).cuda()
        self.net_action_G_B = AGmodel(flag='B2A',opt=self.opt).cuda()
        self.netF_A = Fmodel(self.opt).cuda()

        self.reset_buffer()

        # if self.isTrain:
        self.netD_A = imgDmodel(opt=self.opt).cuda()
        self.netD_B = stateDmodel(opt=self.opt).cuda()
        self.net_action_D_A = ADmodel(opt=self.opt).cuda()
        self.net_action_D_B = ADmodel(opt=self.opt).cuda()

        # if self.isTrain:
        self.fake_A_pool = ImagePool(pool_size=128)
        self.fake_B_pool = ImagePool(pool_size=128)
        self.fake_action_A_pool = ImagePool(pool_size=128)
        self.fake_action_B_pool = ImagePool(pool_size=128)
        # define loss functions
        self.criterionGAN = GANLoss(tensor=self.Tensor).cuda()
        if opt.loss == 'l1':
            self.criterionCycle = nn.L1Loss()
        elif opt.loss == 'l2':
            self.criterionCycle = nn.MSELoss()
        else:
            self.criterionCycle = nn.SmoothL1Loss()
        self.ImgcriterionCycle = nn.MSELoss()
        self.StatecriterionCycle = nn.L1Loss()
        # initialize optimizers
        parameters = [{'params':self.netF_A.parameters(),'lr':self.opt.F_lr},
                     # {'params': self.netF_B.parameters(), 'lr': self.opt.F_lr},
                     # {'params': self.netG_A.parameters(), 'lr': self.opt.G_lr},
                     {'params':self.netG_B.parameters(),'lr':self.opt.G_lr},]
                     # {'params': self.net_action_G_A.parameters(), 'lr': self.opt.A_lr},
                     # {'params': self.net_action_G_B.parameters(), 'lr': self.opt.A_lr}]
        self.optimizer_G = torch.optim.Adam(parameters)
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters())
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters())
        self.optimizer_action_D_A = torch.optim.Adam(self.net_action_D_A.parameters())
        self.optimizer_action_D_B = torch.optim.Adam(self.net_action_D_B.parameters())

        self.use_mask = opt.use_mask
        self.mask = np.array(opt.mask)
        self.mask = torch.tensor(self.mask).float()

        print('---------- Networks initialized ---------------')
        print('-----------------------------------------------')

    def parallel_init(self,device_ids=[0]):
        self.netG_B = torch.nn.DataParallel(self.netG_B,device_ids=device_ids)
        self.netF_A = torch.nn.DataParallel(self.netF_A,device_ids=device_ids)
        self.netD_A = torch.nn.DataParallel(self.netD_A,device_ids=device_ids)
        self.netD_B = torch.nn.DataParallel(self.netD_B,device_ids=device_ids)

    def train_forward_state(self,dataF,pretrained=False):
        if self.use_mask:
            weight_path = os.path.join(self.opt.log_root, '{}_{}_data'.format(self.opt.domain_name, self.opt.task_name),
                                       '{}_{}/pred_mask.pth'.format(self.opt.data_type1, self.opt.data_id1))
        else:
            weight_path = os.path.join(self.opt.log_root, '{}_{}_data'.format(self.opt.domain_name, self.opt.task_name),
                                       '{}_{}/pred.pth'.format(self.opt.data_type1, self.opt.data_id1))
        if pretrained:
            self.netF_A.load_state_dict(torch.load(weight_path))
            print('forward model has loaded!')
            return None
        lr = 1e-3
        optimizer = torch.optim.Adam(self.netF_A.parameters(),lr=lr)
        loss_fn = nn.L1Loss()
        data_size = len(dataF)
        for epoch in range(self.opt.f_epoch):
            epoch_loss, cmp_loss = 0, 0
            if epoch in [3,7,10,15]:
                lr *= 0.5
                optimizer = torch.optim.Adam(self.netF_A.parameters(), lr=lr)
            for i,item in enumerate(tqdm(dataF)):
                if i>data_size*0.8:
                    continue
                state, action, result = item
                state = state.float().cuda()
                action = action.float().cuda()
                result = result.float().cuda()
                out = self.netF_A(state, action)
                if self.use_mask:
                    loss = ((out-result)*(self.mask).cuda()).abs().mean()
                else:
                    loss = loss_fn(out, result)
                # loss = loss_fn(out, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                cmp_loss += loss_fn(state,result).item()
            print('epoch:{} loss:{:.7f} cmp_loss:{:.7f}'
                  .format(epoch,epoch_loss/(0.8*data_size),cmp_loss/(0.8*data_size)))
            torch.save(self.netF_A.state_dict(), weight_path)
        print('forward model has been trained!')

        print('forward model starts to evaluate!')
        epoch_loss, cmp_loss = 0, 0
        for i, item in enumerate(tqdm(dataF)):
            if i<data_size*0.8:
                continue
            state, action, result = item
            state = state.float().cuda()
            action = action.float().cuda()
            result = result.float().cuda()
            out = self.netF_A(state, action)
            loss = loss_fn(out, result)
            epoch_loss += loss.item()
            cmp_loss += loss_fn(state, result).item()
        print('loss:{:.7f} cmp_loss:{:.7f}'.
              format(epoch_loss/(0.2*data_size), cmp_loss/(0.2*data_size)))


    def set_input(self, input):
        # A is state
        self.input_A = input[1][0]

        # B is img
        self.input_Bt0 = input[0][0]
        self.input_Bt1 = input[0][2]
        self.action = input[0][1]
        self.gt0 = input[2][0].float().cuda()
        self.gt1 = input[2][1].float().cuda()


    def forward(self):
        self.real_A = Variable(self.input_A).float().cuda()
        self.real_Bt0 = Variable(self.input_Bt0).float().cuda()
        self.real_Bt1 = Variable(self.input_Bt1).float().cuda()
        self.action = Variable(self.action).float().cuda()


    def test(self):
        # forward
        self.forward()
        # G_A and G_B
        self.backward_G()
        self.backward_D_B()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        if self.isTrain:
            loss_D.backward()
        return loss_D

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_At0.detach())
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()

    def backward_G(self):
        lambda_G_B0 = self.opt.lambda_G0
        lambda_G_B1 = self.opt.lambda_G1
        lambda_G_B2 = self.opt.lambda_G2
        lambda_F = self.opt.lambda_F

        # GAN loss D_B(G_B(B))
        fake_At0 = self.netG_B(self.real_Bt0)
        pred_fake = self.netD_B(fake_At0)
        loss_G_Bt0 = self.criterionGAN(pred_fake, True) * lambda_G_B0

        # GAN loss D_B(G_B(B))
        fake_At1 = self.netF_A(fake_At0,self.action)
        pred_fake = self.netD_B(fake_At1)
        loss_G_Bt1 = self.criterionGAN(pred_fake, True) * lambda_G_B1

        # cycle loss
        pred_At1 = self.netG_B(self.real_Bt1)
        cycle_label = torch.zeros_like(fake_At1).float().cuda()

        if self.use_mask:
            diff = (fake_At1 - pred_At1) * self.mask.cuda(device=fake_At1.device)
        else:
            diff = fake_At1 - pred_At1
        loss_cycle = self.criterionCycle(diff, cycle_label) * lambda_F

        pred_fake = self.netD_B(pred_At1)
        loss_G_Bt2 = self.criterionGAN(pred_fake, True) * lambda_G_B2

        self.loss_state_lt0 = nn.L1Loss()(fake_At0, self.gt0)
        self.loss_state_lt1 = nn.L1Loss()(pred_At1, self.gt1)

        # combined loss
        loss_G = loss_G_Bt0 + loss_G_Bt1 + loss_G_Bt2 + loss_cycle
        # loss_G = self.loss_state_lt0+self.loss_state_lt1


        if self.isTrain:
            loss_G.backward()

        self.fake_At0 = fake_At0.data
        self.fake_At1 = fake_At1.data

        self.loss_G_Bt0 = loss_G_Bt0.item()
        self.loss_G_Bt1 = loss_G_Bt1.item()
        self.loss_cycle = loss_cycle.item()

        self.loss_state_lt0 = self.loss_state_lt0.item()
        self.loss_state_lt1 = self.loss_state_lt1.item()
        self.gt_buffer0.append(self.gt0.cpu().data.numpy())
        self.pred_buffer0.append(self.fake_At0.cpu().data.numpy())
        self.gt_buffer1.append(self.gt1.cpu().data.numpy())
        self.pred_buffer1.append(self.fake_At1.cpu().data.numpy())

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

        self.push_current_errors()

    def push_current_errors(self):
        ret_errors = OrderedDict([('L_t0',self.loss_state_lt0), ('L_t1',self.loss_state_lt1),
                                  ('D_B', self.loss_D_B), ('G_B0', self.loss_G_Bt0),
                                  ('G_B1', self.loss_G_Bt1), ('Cyc',  self.loss_cycle)])
        self.error.append(ret_errors)


    def get_current_errors(self):
        ret_errors = OrderedDict([('L_t0',self.loss_state_lt0), ('L_t1',self.loss_state_lt1),
                                  ('D_B', self.loss_D_B), ('G_B0', self.loss_G_Bt0),
                                  ('G_B1', self.loss_G_Bt1), ('Cyc',  self.loss_cycle)])
        for errors in self.error:
            for key, value in errors.items():
                ret_errors[key] += value
        for key, value in ret_errors.items():
            ret_errors[key] /= (len(self.error)+1)
        self.error = []
        return ret_errors

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, path):
        save_filename = 'model_{}.pth'.format(network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    def save(self, path):
        self.save_network(self.netG_B, 'G_B', path)
        self.save_network(self.netD_B, 'D_B', path)
        self.save_network(self.netG_A, 'G_A', path)
        self.save_network(self.netD_A, 'D_A', path)

        self.save_network(self.net_action_G_B, 'action_G_B', path)
        self.save_network(self.net_action_D_B, 'action_D_B', path)
        self.save_network(self.net_action_G_A, 'action_G_A', path)
        self.save_network(self.net_action_D_A, 'action_D_A', path)

    def load_network(self, network, network_label, path):
        weight_filename = 'model_{}.pth'.format(network_label)
        weight_path = os.path.join(path, weight_filename)
        network.load_state_dict(torch.load(weight_path))

    def load(self,path):
        self.load_network(self.netG_B, 'G_B', path)
        self.load_network(self.netD_B, 'D_B', path)
        self.load_network(self.netG_A, 'G_A', path)
        self.load_network(self.netD_A, 'D_A', path)

        self.load_network(self.net_action_G_B, 'action_G_B', path)
        self.load_network(self.net_action_D_B, 'action_D_B', path)
        self.load_network(self.net_action_G_A, 'action_G_A', path)
        self.load_network(self.net_action_D_A, 'action_D_A', path)

    def show_points(self,gt_data,pred_data):
        print(abs(gt_data-pred_data).mean(0))
        ncols = int(np.sqrt(gt_data.shape[1]))+1
        nrows = int(np.sqrt(gt_data.shape[1]))+1
        assert (ncols*nrows>=gt_data.shape[1])
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i>=gt_data.shape[1]:
                continue
            ax.scatter(gt_data[:, ax_i], pred_data[:, ax_i], s=3, label='xyz_{}'.format(ax_i))


    def npdata(self,item):
        return item.cpu().data.numpy()

    def reset_buffer(self):
        self.gt_buffer0 = []
        self.pred_buffer0 = []
        self.gt_buffer1 = []
        self.pred_buffer1 = []
        self.error = []


    def visual(self,path):
        gt_data = np.vstack(self.gt_buffer0)
        pred_data = np.vstack(self.pred_buffer0)
        self.show_points(gt_data,pred_data)
        # plt.legend()
        plt.savefig(path)
        plt.cla()
        plt.clf()

        gt_data = np.vstack(self.gt_buffer1)
        pred_data = np.vstack(self.pred_buffer1)
        self.show_points(gt_data, pred_data)
        # plt.legend()
        plt.savefig(path.replace('.jpg','_step1.jpg'))
        self.reset_buffer()


if __name__ == '__main__':
    mymodel = CycleGANModel()
    print(mymodel)