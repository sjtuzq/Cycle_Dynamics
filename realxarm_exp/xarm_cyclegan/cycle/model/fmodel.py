
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms

import sys
sys.path.append('../')

class Fmodel(nn.Module):
    def __init__(self,opt):
        super(Fmodel,self).__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.statefc = nn.Sequential(
            nn.Linear(self.state_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
        )
        self.actionfc = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.predfc = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,self.state_dim)
        )

    def forward(self, state,action):
        state_feature = self.statefc(state)
        action_feature = self.actionfc(action)
        feature = torch.cat((state_feature,action_feature),1)
        return self.predfc(feature)

class AGmodel(nn.Module):
    def __init__(self,flag='A2B',opt=None):
        super(AGmodel,self).__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.action_fix = opt.action_fix
        self.stn = nn.Sequential(
            nn.Linear(self.state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.state_dim**2)
        )
        self.actionfc = nn.Sequential(
            nn.Linear(self.action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )

    def forward2(self, action):
        feature = torch.ones_like(action).float().cuda()
        trans = self.stn(feature).view(-1,2,2)
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 1]).astype(np.float32))).\
                          view(1, 4).repeat(action.shape[0], 1).view(-1,2,2).cuda()
        trans = trans + iden
        # trans = self.trans.repeat(action.shape[0], 1).view(-1,2,2).cuda()

        action = torch.bmm(trans,action.unsqueeze(-1))
        return action.squeeze(-1)

    def forward1(self, action):
        return self.actionfc(action)

    def forward0(self, action):
        return action

    def forward(self, action):
        if self.action_fix:
            return self.forward0(action)
        else:
            return self.forward1(action)

class ADmodel(nn.Module):
    def __init__(self,opt=None):
        super(ADmodel,self).__init__()
        self.opt = opt
        self.action_dim = opt.action_dim
        self.actionfc = nn.Sequential(
            nn.Linear(self.action_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, action):
        return self.actionfc(action)

class ImgFmodel(nn.Module):
    def __init__(self,opt=None):
        super(ImgFmodel, self).__init__()
        self.opt = opt
        self.ngf = 32
        self.inc = Inconv(3, self.ngf)
        self.downnet = nn.Sequential(
            Down(32,64),
            Down(64,128),
            Down(128,128),
            Down(128,256),
            Down(256,512),
        )
        self.action_dim = opt.action_dim
        self.action_feature = self.ngf * 4
        self.upnet = nn.Sequential(
            Up(512 + self.action_feature,256),
            Up(256,128),
            Up(128,128),
            Up(128,64),
            Up(64,32)
        )
        self.outc = Outconv(self.ngf, 3)
        self.action_fc = nn.Sequential(
            nn.Linear(in_features=self.action_dim,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=self.action_feature),
            nn.ReLU()
        )

    def forward(self, imgA,action):
        input = self.inc(imgA)
        feature = self.downnet(input)
        action_f = self.action_fc(action)
        action_f = action_f.repeat(8,8,1,1).permute(2,3,0,1)

        feature = torch.cat((feature,action_f),1)

        out = self.upnet(feature)
        return self.outc(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            #norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x

class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                      bias=use_bias),
            #norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            #norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


def train_imgf(opt):
    model = ImgFmodel().cuda()
    dataset = Robotdata.get_loader(opt)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        for i,item in enumerate(dataset):
            state,action,result = item[0]
            state = state.float().cuda()
            action = action.float().cuda()
            result = result.float().cuda()

            out = model(state,action)
            loss = loss_fn(out,result)*100
            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(),'./imgpred.pth')

        # generate the batch image
        merge_img = []
        for (before, after, pred) in zip(state, result, out):
            img = torch.cat([before, after, pred], 1)
            merge_img.append(img)
        merge_img = torch.cat(merge_img, 2).cpu()
        merge_img = (merge_img + 1) / 2
        img = transforms.ToPILImage()(merge_img)
        # img = transforms.Resize((512, 640))(img)
        img.save(os.path.join('./tmp/imgpred', 'img_{}.jpg'.format(epoch)))

def train_statef(opt):
    model = Fmodel(opt).cuda()
    weight_path = os.path.join(opt.data_root, 'data_{}/pred.pth'.format(opt.test_id1))
    # model.load_state_dict(torch.load(weight_path))
    # dataset = Robotdata.get_loader(opt)
    dataset = RobotStackFdata.get_loader(opt)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.L1Loss()

    for epoch in range(50):
        epoch_loss,cmp_loss = 0,0
        for i, item in enumerate(tqdm(dataset)):
            state, action, result = item
            state = state.float().cuda()
            action = action.float().cuda()[:,:opt.action_dim]
            result = result.float().cuda()
            out = model(state, action)
            loss = loss_fn(out, result)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            cmp_loss += loss_fn(state,result).item()
        print('epoch:{} loss:{:.7f} cmp:{:.7f}'.format(epoch, epoch_loss / len(dataset),cmp_loss / len(dataset)))
        weight_path = os.path.join(opt.data_root,'data_{}/pred.pth'.format(opt.test_id1))
        torch.save(model.state_dict(),weight_path)

def eval_statef(opt):
    model = Fmodel(opt).cuda()
    weight_path = os.path.join(opt.data_root, 'data_{}/pred.pth'.format(opt.test_id1))
    model.load_state_dict(torch.load(weight_path))
    # dataset = Robotdata.get_loader(opt)
    dataset = RobotStackFdata.get_loader(opt)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.L1Loss()

    epoch_loss,cmp_loss = 0,0
    for i, item in enumerate(tqdm(dataset)):
        state, action, result = item
        state = state.float().cuda()
        action = action.float().cuda()[:,:opt.action_dim]
        result = result.float().cuda()
        out = model(state, action)
        loss = loss_fn(out, result)
        epoch_loss += loss.item()
        cmp_loss += loss_fn(state,result).item()
    print('loss:{:.7f} cmp:{:.7f}'.format(epoch_loss / len(dataset),cmp_loss / len(dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--exp_id', type=int, default=1, help='experiment id')
    parser.add_argument('--test_id1', type=int, default=10, help='dataset test id')
    parser.add_argument('--test_id2', type=int, default=10, help='dataset test id')

    # fetch series: state_dim=25, action_dim=4
    # control series: state_dim=4, action_dim=1
    # walk series: state_dim=17, action_dim=6
    # reacher: state_dim=11, action_dim=2
    parser.add_argument('--state_dim', type=int, default=10, help='state dimension')
    parser.add_argument('--action_dim', type=int, default=3, help='action dimension')
    parser.add_argument('--stack_n', type=int, default=3, help='action dimension')
    parser.add_argument('--img_size', type=int, default=84, help='action dimension')


    parser.add_argument('--loss', type=str, default='l2', help='loss function')
    parser.add_argument('--imgA_id', type=int, default=1, help='datasetA from view1')
    parser.add_argument('--imgB_id',type=int,default=2,help='datasetB from view2')
    parser.add_argument('--data_root', type=str, default='../../../logs', help='logs root path')

    opt = parser.parse_args()
    train_statef(opt)

    opt.istrain = False
    with torch.no_grad():
        eval_statef(opt)

