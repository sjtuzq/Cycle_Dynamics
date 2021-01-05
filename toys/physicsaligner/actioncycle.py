

import os
import torch
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools


from neuralmodel import img2state,state2img,imgDmodel,stateDmodel
from physicsengine import CDFdata
from utils import ImagePool,GANLoss
from fmodel import Fmodel,Amodel


class CycleActionModel():
    def __init__(self,opt):
        self.opt = opt
        self.dynamic = opt.dynamic
        self.isTrain = opt.istrain
        self.Tensor = torch.cuda.FloatTensor

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_B = img2state().cuda()
        self.netF_A = Fmodel().cuda()
        self.netAction = Amodel().cuda()
        self.dataF = CDFdata.get_loader(opt)
        self.train_forward(pretrained=True)

        self.gt_buffer = []
        self.pred_buffer = []

        # if self.isTrain:
        self.netD_B = stateDmodel().cuda()

        # if self.isTrain:
        self.fake_A_pool = ImagePool(pool_size=128)
        self.fake_B_pool = ImagePool(pool_size=128)
        # define loss functions
        self.criterionGAN = GANLoss(tensor=self.Tensor).cuda()
        if opt.loss == 'l1':
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
        elif opt.loss == 'l2':
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionIdt = torch.nn.MSELoss()
        # initialize optimizers
        # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
        self.optimizer_G = torch.optim.Adam([{'params':self.netF_A.parameters(),'lr':self.opt.F_lr},
                                             {'params':self.netG_B.parameters(),'lr':self.opt.G_lr},
                                             {'params': self.netAction.parameters(), 'lr': self.opt.G_lr}])
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters())


        print('---------- Networks initialized ---------------')
        print('-----------------------------------------------')


    def train_forward(self,pretrained=False):
        if pretrained:
            self.netF_A.load_state_dict(torch.load('./pred.pth'))
            return None
        optimizer = torch.optim.Adam(self.netF_A.parameters(),lr=1e-3)
        loss_fn = torch.nn.L1Loss()
        for epoch in range(10):
            epoch_loss = 0
            for i,item in enumerate(tqdm(self.dataF)):
                state, action, result = item[1]
                state = state.float().cuda()
                action = action.float().cuda()
                result = result.float().cuda()
                out = self.netF_A(state, action)
                loss = loss_fn(out, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print('epoch:{} loss:{:.7f}'.format(epoch,epoch_loss/len(self.dataF)))
            torch.save(self.netF_A.state_dict(), './pred.pth')
        print('forward model has been trained!')


    def set_input(self, input):
        # AtoB = self.opt.which_direction == 'AtoB'
        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A']
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
        fake_A = self.fake_A_pool.query(self.fake_At0)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()

    def backward_G(self):
        lambda_G_B0 = 1.0
        lambda_G_B1 = 1.0
        lambda_F = self.opt.lambda_F


        # GAN loss D_B(G_B(B))
        fake_At0 = self.netG_B(self.real_Bt0)
        pred_fake = self.netD_B(fake_At0)
        loss_G_Bt0 = self.criterionGAN(pred_fake, True) * lambda_G_B0

        # GAN loss D_B(G_B(B))
        fake_action = self.netAction(self.action)
        fake_At1 = self.netF_A(fake_At0,fake_action)

        # fake_At1 = self.netF_A(fake_At0,self.action)

        pred_fake = self.netD_B(fake_At1)
        loss_G_Bt1 = self.criterionGAN(pred_fake, True) * lambda_G_B1

        # cycle loss
        pred_At1 = self.netG_B(self.real_Bt1)
        cycle_label = torch.zeros_like(fake_At1).float().cuda()
        loss_cycle = self.criterionCycle(fake_At1-pred_At1,cycle_label) * lambda_F

        # combined loss
        loss_G = loss_G_Bt0 + loss_G_Bt1 + loss_cycle
        if self.isTrain:
            loss_G.backward()


        self.fake_At0 = fake_At0.data
        self.fake_At1 = fake_At1.data


        self.loss_G_Bt0 = loss_G_Bt0.item()
        self.loss_G_Bt1 = loss_G_Bt1.item()
        self.loss_cycle = loss_cycle.item()

        self.loss_state_lt0 = self.criterionCycle(self.fake_At0, self.gt0).item()
        self.loss_state_lt1 = self.criterionCycle(self.fake_At1, self.gt1).item()
        self.gt_buffer.append(self.gt0.cpu().data.numpy())
        self.gt_buffer.append(self.gt1.cpu().data.numpy())
        self.pred_buffer.append(self.fake_At0.cpu().data.numpy())
        self.pred_buffer.append(self.fake_At1.cpu().data.numpy())

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

    def get_current_errors(self):
        ret_errors = OrderedDict([('L_t0',self.loss_state_lt0), ('L_t1',self.loss_state_lt1),
                                  ('D_B', self.loss_D_B), ('G_B0', self.loss_G_Bt0),
                                  ('G_B1', self.loss_G_Bt1), ('Cyc',  self.loss_cycle)])
        # if self.opt.identity > 0.0:
        #     ret_errors['idt_A'] = self.loss_idt_A
        #     ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, path):
        save_filename = 'model_{}.pth'.format(network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    def save(self, path):
        self.save_network(self.netG_B, 'G_B', path)
        self.save_network(self.netD_B, 'D_B', path)

    def load_network(self, network, network_label, path):
        weight_filename = 'model_{}.pth'.format(network_label)
        weight_path = os.path.join(path, weight_filename)
        network.load_state_dict(torch.load(weight_path))

    def load(self,path):
        self.load_network(self.netG_B, 'G_B', path)

    def show_points(self):
        # num_images = min(imgs.shape[0],num_images)
        ncols = 1
        nrows = 2
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()
        gt_data = np.vstack(self.gt_buffer)
        pred_data = np.vstack(self.pred_buffer)
        print(abs(gt_data-pred_data).mean(0))


        for ax_i, ax in enumerate(axes):
            if ax_i < nrows:
                ax.scatter(gt_data[:,ax_i],pred_data[:,ax_i],s=3,label='xyz_{}'.format(ax_i))
            else:
                ax.scatter(self.npdata(self.fake_At1[:, ax_i-nrows]), self.npdata(self.gt1[:, ax_i-nrows]),label='t1_{}'.format(ax_i-nrows))

    def npdata(self,item):
        return item.cpu().data.numpy()

    def visual(self,path):
        # plt.xlim(-4,4)
        # plt.ylim(-1.5,1.5)
        self.show_points()
        plt.legend()
        plt.savefig(path)
        plt.cla()
        plt.clf()
        self.gt_buffer = []
        self.pred_buffer = []


if __name__ == '__main__':
    mymodel = CycleGANModel()
    print(mymodel)