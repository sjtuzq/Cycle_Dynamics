
"""
This file provides models
including Generator and discriminator

"""


import torch
import torch.nn as nn


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            # nn.ReflectionPad2d(2),
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.down(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.down = nn.Sequential(
            # nn.ReflectionPad2d(2),
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        return x


class GModel(nn.Module):
    def __init__(self,opt):
        super(GModel,self).__init__()
        feature_dim = 512
        self.opt = opt

        self.encoder = nn.Sequential(
            Down(3,32),
            Down(32,64),
            Down(64,128),
            Down(128,256),
            Down(256,256),
            Down(256,feature_dim)
        )

        self.decoder = nn.Sequential(
            Up(feature_dim,256),
            Up(256,256),
            Up(256,128),
            Up(128,64),
            Up(64,32),
            Up(32,12),
        )

        self.outc = OutConv(12,3)


    def forward(self, data):
        feature = self.encoder(data)
        # B,F,W,H = feature.shape
        # feature = feature.view(B,int(F/4),2,2)
        output = self.decoder(feature)
        output = self.outc(output)
        return output.squeeze()


class DModel(nn.Module):
    def __init__(self,opt):
        super(DModel,self).__init__()
        self.opt = opt
        feature_dim = 512

        self.encoder = nn.Sequential(
            Down(3,32),
            Down(32,64),
            Down(64,128),
            Down(128,256),
            Down(256,256),
            Down(256,256),
            Down(256,256),
            Down(256,feature_dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(feature_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )

    def forward(self, data):
        feature = self.encoder(data)
        pred = self.fc(feature.squeeze())
        return pred

if __name__ == '__main__':
    model = GModel().cuda()
    data = torch.rand(32,64,64).cuda()
    output = model(data)
    print(output.shape)

    model = DModel().cuda()
    data = torch.rand(32,64,64).cuda()
    output = model(data)
    print(output.shape)