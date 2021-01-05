

import os
import torch
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools


from cycle.models import S2S,SDmodel,AGmodel,ADmodel,Fengine
from cycle.utils import ImagePool,GANLoss
from cycle.crosspolicy import CrossPolicy

class CycleGANModel():
    def __init__(self,opt):
        self.opt = opt
        self.isTrain = opt.istrain
        self.Tensor = torch.cuda.FloatTensor

        self.cross_policy = CrossPolicy(opt)
        self.netG_B = S2S(opt).cuda()
        self.net_action_G_A = AGmodel(opt,'1to2').cuda()
        self.net_action_G_B = AGmodel(opt,'2to1').cuda()
        self.fengine = Fengine(opt)
        self.netF_A = self.fengine.fmodel.cuda()
        self.net_action_D_A = ADmodel(opt, '1to2').cuda()
        self.net_action_D_B = ADmodel(opt, '2to1').cuda()

        self.reset_buffer()
        self.netD_B = SDmodel(opt).cuda()



        self.fake_A_pool = ImagePool(pool_size=128)
        self.fake_B_pool = ImagePool(pool_size=128)
        self.fake_action_A_pool = ImagePool(pool_size=128)
        self.fake_action_B_pool = ImagePool(pool_size=128)
        # define loss functions
        self.criterionGAN = GANLoss(tensor=self.Tensor).cuda()
        if opt.loss == 'l1':
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
        elif opt.loss == 'l2':
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionIdt = torch.nn.MSELoss()
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam([{'params':self.netF_A.parameters(),'lr':0.0},
                                             {'params': self.net_action_G_A.parameters(), 'lr': opt.lr_Ax},
                                             {'params': self.net_action_G_B.parameters(), 'lr': opt.lr_Ax},
                                             {'params':self.netG_B.parameters(),'lr':opt.lr_Gx}])
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters())
        self.optimizer_D_action_B = torch.optim.Adam(self.net_action_D_B.parameters())
        self.optimizer_D_action_A = torch.optim.Adam(self.net_action_D_A.parameters())
        self.init_start = True

        print('-----------------------------------------------')
        print('---------- Networks initialized ---------------')
        print('-----------------------------------------------')

    def update(self,opt):
        self.init_start = opt.init_start
        self.net_action_G_A.init_start = opt.init_start
        self.net_action_G_B.init_start = opt.init_start
        self.optimizer_G = torch.optim.Adam([{'params': self.netF_A.parameters(), 'lr': 0.0},
                                             {'params': self.net_action_G_A.parameters(), 'lr': opt.lr_Ax},
                                             {'params': self.net_action_G_B.parameters(), 'lr': opt.lr_Ax},
                                             {'params': self.netG_B.parameters(), 'lr': opt.lr_Gx}])
        print('\n-----------------------------------------------')
        print('------------ model phase updated! -------------')
        print('-----------------------------------------------\n')

    def set_input(self, input):
        data1,data2 = input
        self.input_At0 = self.Tensor(data1[0])
        self.action_A = self.Tensor(data1[1])
        self.input_At1 = self.Tensor(data1[2])

        self.input_Bt0 = self.Tensor(data2[0])
        self.action_B = self.Tensor(data2[1])
        self.input_Bt1 = self.Tensor(data2[2])

    def forward(self):
        self.real_At0 = Variable(self.input_At0)
        self.action_A = Variable(self.action_A)
        self.real_At1 = Variable(self.input_At1)

        self.real_Bt0 = Variable(self.input_Bt0)
        self.action_B = Variable(self.action_B)
        self.real_Bt1 = Variable(self.input_Bt1)

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
        fake_A = self.fake_A_pool.query(self.fake_At0)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_At0, fake_A)
        self.loss_D_B = loss_D_B.item()

    def backward_D_action_B(self):
        fake_A = self.fake_action_A_pool.query(self.fake_action_A)
        loss_D_action_B = self.backward_D_basic(self.net_action_D_B, self.action_A, fake_A)
        self.loss_D_action_B = loss_D_action_B.item()

    def backward_D_action_A(self):
        fake_B = self.fake_action_B_pool.query(self.fake_action_B)
        loss_D_action_A = self.backward_D_basic(self.net_action_D_A, self.action_B, fake_B)
        self.loss_D_action_A = loss_D_action_A.item()

    def backward_G(self):
        lambda_G_B0 = self.opt.lambda_G0
        lambda_G_B1 = self.opt.lambda_G1
        lambda_G_action_A = self.opt.lambda_GactA
        lambda_G_action_B = self.opt.lambda_GactB
        lambda_cycle_action = self.opt.lambda_Gcyc
        lambda_F = self.opt.lambda_F
        
        # ***************************
        #       action cycle part
        # ***************************

        # action cycle of B
        fake_action_A = self.net_action_G_B(self.action_B)
        back_action_B = self.net_action_G_A(fake_action_A)

        pred_fake_action_A = self.net_action_D_B(fake_action_A)
        loss_G_action_B = self.criterionGAN(pred_fake_action_A, True) * lambda_G_action_B
        loss_action_cycle_B = self.criterionCycle(back_action_B,self.action_B) * lambda_cycle_action


        # action cycle of A
        fake_action_B = self.net_action_G_A(self.action_A)
        back_action_A = self.net_action_G_B(fake_action_B)

        pred_fake_action_B = self.net_action_D_A(fake_action_B)
        loss_G_action_A = self.criterionGAN(pred_fake_action_B, True) * lambda_G_action_A
        loss_action_cycle_A = self.criterionCycle(back_action_A, self.action_A) * lambda_cycle_action

        loss_action_cycle = loss_G_action_B+loss_action_cycle_B+loss_G_action_A+loss_action_cycle_A


        # ***************************
        #       state cycle part
        # ***************************
        
        # GAN loss D_B(G_B(B))
        fake_At0 = self.netG_B(self.real_Bt0)
        pred_fake = self.netD_B(fake_At0)
        loss_G_Bt0 = self.criterionGAN(pred_fake, True) * lambda_G_B0

        # GAN loss D_B(G_B(B))
        fake_At1 = self.netF_A(fake_At0,fake_action_A)
        pred_fake = self.netD_B(fake_At1)
        loss_G_Bt1 = self.criterionGAN(pred_fake, True) * lambda_G_B1

        # cycle loss
        pred_At1 = self.netG_B(self.real_Bt1)
        cycle_label = torch.zeros_like(fake_At1).float().cuda()
        loss_cycle = self.criterionCycle(fake_At1-pred_At1,cycle_label) * lambda_F

        # ***************************
        #       loss backward part
        # ***************************

        # combined loss
        loss_G = loss_G_Bt0 + loss_G_Bt1 + loss_cycle
        loss = loss_G + loss_action_cycle

        if self.isTrain:
            loss.backward()

        # ***************************
        #      postprocess part
        # ***************************

        self.fake_At0 = fake_At0.data
        self.fake_At1 = fake_At1.data
        self.fake_action_A = fake_action_A
        self.fake_action_B = fake_action_B

        self.loss_G_Bt0 = loss_G_Bt0.item()
        self.loss_G_Bt1 = loss_G_Bt1.item()
        self.loss_cycle = loss_cycle.item()
        self.loss_G_action_B = loss_G_action_B.item()
        self.loss_G_action_A = loss_G_action_A.item()
        self.loss_action_cycle_A = loss_action_cycle_A.item()
        self.loss_action_cycle_B = loss_action_cycle_B.item()

        self.loss_state_lt0,self.loss_state_lt1 = 0,0

        self.gt0 = self.real_Bt0
        self.gt1 = self.real_Bt1
        self.gt_buffer.append(self.gt0.cpu().data.numpy())
        self.gt_buffer1.append(self.gt1.cpu().data.numpy())
        self.pred_buffer.append(self.fake_At0.cpu().data.numpy())
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
        # D_action_A
        self.optimizer_D_action_B.zero_grad()
        self.backward_D_action_B()
        self.optimizer_D_action_B.step()
        # D_action_B
        self.optimizer_D_action_A.zero_grad()
        self.backward_D_action_A()
        self.optimizer_D_action_A.step()

    def fetch(self):
        real_B = (self.real_Bt0.cpu().data.numpy(),
                  self.action_B.cpu().data.numpy(),
                  self.real_Bt1.cpu().data.numpy())
        fake_B = (self.fake_At0.cpu().data.numpy(),
                  self.action_B.cpu().data.numpy(),
                  self.fake_At1.cpu().data.numpy())
        return (real_B,fake_B)

    def get_current_errors(self):
        ret_errors = OrderedDict([
            # ('L_t0',self.loss_state_lt0), ('L_t1',self.loss_state_lt1),
            ('D_B', self.loss_D_B), ('G_B0', self.loss_G_Bt0),
            ('G_B1', self.loss_G_Bt1), ('Cyc', self.loss_cycle),
            ('D_act_B', self.loss_D_action_B), ('G_act_B', self.loss_G_action_B),
            ('D_act_A', self.loss_D_action_A), ('G_act_A', self.loss_G_action_A),
            ('Cyc_act_A', self.loss_action_cycle_A), ('Cyc_act_B', self.loss_action_cycle_B),
        ])
        return ret_errors

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, path):
        save_filename = 'model_{}.pth'.format(network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    def save(self, path):
        self.save_network(self.netG_B, 'G_B', path)
        self.save_network(self.netD_B, 'D_B', path)
        self.save_network(self.net_action_G_B, 'G_act_B', path)
        self.save_network(self.net_action_G_A, 'G_act_A', path)
        self.save_network(self.net_action_D_A, 'D_act_A', path)
        self.save_network(self.net_action_D_B, 'D_act_B', path)

    def load_network(self, network, network_label, path):
        weight_filename = 'model_{}.pth'.format(network_label)
        weight_path = os.path.join(path, weight_filename)
        network.load_state_dict(torch.load(weight_path))

    def load(self,path):
        self.load_network(self.netG_B, 'G_B', path)
        self.load_network(self.netD_B, 'D_B', path)
        self.load_network(self.net_action_G_B, 'G_act_B', path)
        self.load_network(self.net_action_G_A, 'G_act_A', path)
        self.load_network(self.net_action_D_A, 'D_act_A', path)
        self.load_network(self.net_action_D_B, 'D_act_B', path)


    def reset_buffer(self):
        self.gt_buffer = []
        self.pred_buffer = []
        self.gt_buffer1 = []
        self.pred_buffer1 = []

    def show_points(self,gt_data,pred_data):
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

    def visual(self,path):
        gt_data = np.vstack(self.gt_buffer)
        pred_data = np.vstack(self.pred_buffer)
        # self.show_points(gt_data,pred_data)
        # plt.savefig(path)
        # plt.cla()
        # plt.clf()

        # gt_data = np.vstack(self.gt_buffer1)
        # pred_data = np.vstack(self.pred_buffer1)
        # print(list(abs(gt_data - pred_data).mean(0)), abs(gt_data - pred_data).mean())
        self.reset_buffer()
