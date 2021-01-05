
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from physicsengine import CDFdata

class Fmodel(nn.Module):
    def __init__(self):
        super(Fmodel,self).__init__()
        self.statefc = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,32)
        )
        self.actionfc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        self.predfc = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )

    def forward(self, state,action):
        state_feature = self.statefc(state)
        action_feature = self.actionfc(action)
        feature = torch.cat((state_feature,action_feature),1)
        return self.predfc(feature)

class AGmodel(nn.Module):
    def __init__(self,flag='A2B'):
        super(AGmodel,self).__init__()
        self.flag = flag
        if self.flag == 'B2A':
            self.trans = Variable(torch.from_numpy(np.array([1/0.6, 0, 0, 1/0.6]).
                                                   astype(np.float32))).view(1, 4)
        elif self.flag == 'A2B':
            self.trans = Variable(torch.from_numpy(np.array([1*0.6, 0, 0, 1*0.6]).
                                                   astype(np.float32))).view(1, 4)
        self.stn = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        self.actionfc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
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
        # return action
        return self.actionfc(action)

    def forward0(self, action):
        return action

    def forward(self, action):
        return self.forward1(action)

class ADmodel(nn.Module):
    def __init__(self):
        super(ADmodel,self).__init__()
        self.actionfc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, action):
        return self.actionfc(action)

class ImgFmodel(nn.Module):
    def __init__(self):
        super(ImgFmodel, self).__init__()
        self.ngf = 32
        self.inc = Inconv(3, self.ngf)
        self.downnet = nn.Sequential(
            Down(32,64),
            Down(64,128),
            Down(128,128),
            Down(128,256),
            Down(256,512),
        )
        self.action_d = self.ngf * 4
        self.upnet = nn.Sequential(
            Up(512 + self.action_d,256),
            Up(256,128),
            Up(128,128),
            Up(128,64),
            Up(64,32)
        )
        self.outc = Outconv(self.ngf, 3)
        self.action_fc = nn.Sequential(
            nn.Linear(in_features=2,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=self.action_d),
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
            norm_layer(out_ch),
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
            norm_layer(out_ch),
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
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--exp_id', type=int, default=1, help='experiment id')
    parser.add_argument('--test_id1', type=int, default=1, help='dataset test id')
    parser.add_argument('--test_id2', type=int, default=4, help='dataset test id')
    parser.add_argument('--dynamic',type=bool,default=True,help='whether to use dynamic model')
    parser.add_argument('--clockwise', type=bool, default=False, help='dynamic direction')
    parser.add_argument('--domain',type=str,default='circle',help='the distribution of points')

    parser.add_argument('--loss', type=str, default='l2', help='loss function')
    parser.add_argument('--imgA_id', type=int, default=1, help='datasetA from view1')
    parser.add_argument('--imgB_id',type=int,default=2,help='datasetB from view2')
    parser.add_argument('--data_root', type=str, default='./tmp/data', help='logs root path')

    opt = parser.parse_args()

    model = ImgFmodel().cuda()
    dataset = CDFdata.get_loader(opt)
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