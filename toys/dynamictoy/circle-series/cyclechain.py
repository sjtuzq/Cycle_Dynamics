

import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools


from model import GModel,DModel
from utils import ImagePool,GANLoss
from fmodel import Fmodel


class CycleGANModel():
    def __init__(self,opt):
        self.opt = opt
        self.dynamic = opt.dynamic
        self.isTrain = opt.istrain
        self.Tensor = torch.cuda.FloatTensor

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = GModel(opt).cuda()
        self.netG_B = GModel(opt).cuda()
        self.netF_A = Fmodel().cuda()

        if self.isTrain:
            self.netD_A = DModel(opt).cuda()
            self.netD_B = DModel(opt).cuda()

        if self.isTrain:
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
            self.optimizer_G = torch.optim.Adam([{'params':self.netG_A.parameters(),'lr':1e-3},
                                                 {'params':self.netF_A.parameters(),'lr':0.0},
                                                 {'params':self.netG_B.parameters(),'lr':1e-3}])
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters())
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters())
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                # self.schedulers.append(networks.get_scheduler(optimizer, opt))
                self.schedulers.append(optimizer)

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        # if self.isTrain:
        #     networks.print_network(self.netD_A)
        #     networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        # AtoB = self.opt.which_direction == 'AtoB'
        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A']
        self.input_A = input[0]
        self.input_Bt0 = input[1][0]
        self.input_Bt1 = input[1][2]
        self.action = input[1][1]


    def forward(self):
        self.real_A = Variable(self.input_A).float().cuda()
        self.real_Bt0 = Variable(self.input_Bt0).float().cuda()
        self.real_Bt1 = Variable(self.input_Bt1).float().cuda()
        self.action = Variable(self.action).float().cuda()


    def test(self):
        self.forward()
        real_A = Variable(self.input_A, volatile=True).float().cuda()
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_Bt0, volatile=True).float().cuda()
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data


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
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_Bt0, fake_B)
        self.loss_D_A = loss_D_A.item()

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_At0)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()

    def backward_G(self):
        lambda_idt = 0.5
        lambda_A = 100.0
        lambda_B = 100.0
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_Bt0)
            loss_idt_A = self.criterionIdt(idt_A, self.real_Bt0) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.item()
            self.loss_idt_B = loss_idt_B.item()
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        lambda_G = 1.0

        # --------first cycle-----------#
        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True) * lambda_G
        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # ---------second cycle---------#
        # GAN loss D_B(G_B(B))
        fake_At0 = self.netG_B(self.real_Bt0)
        pred_fake = self.netD_B(fake_At0)
        loss_G_Bt0 = self.criterionGAN(pred_fake, True) * lambda_G
        # Backward cycle loss
        rec_Bt0 = self.netG_A(fake_At0)
        loss_cycle_Bt0 = self.criterionCycle(rec_Bt0, self.real_Bt0) * lambda_B

        # ---------third cycle---------#
        # GAN loss D_B(G_B(B))
        fake_At1 = self.netF_A(fake_At0,self.action)
        pred_fake = self.netD_B(fake_At1)
        loss_G_Bt1 = self.criterionGAN(pred_fake, True) * lambda_G
        # Backward cycle loss
        rec_Bt1 = self.netG_A(fake_At1)
        loss_cycle_Bt1 = self.criterionCycle(rec_Bt1, self.real_Bt1) * lambda_B


        # combined loss
        loss_G = loss_idt_A + loss_idt_B
        loss_G = loss_G + loss_G_A + loss_cycle_A
        loss_G = loss_G + loss_G_Bt0 + loss_cycle_Bt0
        if self.dynamic:
            loss_G = loss_G + loss_G_Bt1 + loss_cycle_Bt1
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_At0 = fake_At0.data
        self.fake_At1 = fake_At1.data
        self.rec_A = rec_A.data
        self.rec_Bt0 = rec_Bt0.data
        self.rec_Bt1 = rec_Bt1.data

        self.loss_G_A = loss_G_A.item()
        self.loss_G_Bt0 = loss_G_Bt0.item()
        self.loss_G_Bt1 = loss_G_Bt1.item()
        self.loss_cycle_A = loss_cycle_A.item()
        self.loss_cycle_Bt0 = loss_cycle_Bt0.item()
        self.loss_cycle_Bt1 = loss_cycle_Bt1.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A),
                                  ('Cyc_A', self.loss_cycle_A), ('D_B', self.loss_D_B),
                                  ('G_B', self.loss_G_Bt0), ('Cyc_B',  self.loss_cycle_Bt0)])
        # if self.opt.identity > 0.0:
        ret_errors['idt_A'] = self.loss_idt_A
        ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, path):
        save_filename = 'model_{}.pth'.format(network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    def save(self, path):
        self.save_network(self.netG_A, 'G_A', path)
        self.save_network(self.netD_A, 'D_A', path)
        self.save_network(self.netG_B, 'G_B', path)
        self.save_network(self.netD_B, 'D_B', path)

    def load_network(self, network, network_label, path):
        weight_filename = 'model_{}.pth'.format(network_label)
        weight_path = os.path.join(path, weight_filename)
        network.load_state_dict(torch.load(weight_path))

    def load(self,path):
        self.load_network(self.netG_A, 'G_A', path)
        self.load_network(self.netG_B, 'G_B', path)

    def plot_points(self,item,label):
        item = item.cpu().data.numpy()
        plt.scatter(item[:,0],item[:,1],label=label)

    def visual(self,path):
        plt.xlim(-4,4)
        plt.ylim(-1.5,1.5)
        self.plot_points(self.real_A,'realA')
        self.plot_points(self.fake_B,'fake_B')
        self.plot_points(self.rec_A,'rec_A')
        # self.plot_points(self.real_B,'real_B')
        for p1,p2 in zip(self.real_A,self.fake_B):
            p1,p2 = p1.cpu().data.numpy(),p2.cpu().data.numpy()
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]])
        plt.legend()
        plt.savefig(path)
        plt.cla()
        plt.clf()


if __name__ == '__main__':
    mymodel = CycleGANModel()
    print(mymodel)