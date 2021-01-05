

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

from cycle.model.utils import init_net,get_norm_layer
from cycle.model.layers import Up,Down,Inconv,Outconv

class state2state(nn.Module):
    def __init__(self,opt):
        super(state2state,self).__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.state_dim)
        )

    def forward(self, state):
        return self.fc(state)


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.up1 = Up(96,96)
        self.up2 = Up(96,48)
        self.up3 = Up(48,24)
        self.up4 = Up(24,12)
        self.up5 = Up(12,6)
        self.up6 = Up(6,6)

        self.outc = Outconv(6,3)


    def forward(self, input):
        out = self.up1(input)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)
        out = self.up5(out)
        out = self.up6(out)

        return self.outc(out)


def state2img(opt=None):
    init_type='normal'
    init_gain=0.02
    gpu_id='cuda:0'
    net = ImgGenerator(opt=opt)

    return init_net(net, init_type, init_gain, gpu_id)

class ImgGenerator(nn.Module):
    def __init__(self, opt=None,input_nc=3, output_nc=3, ngf=32,
                 n_down=6, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ImgGenerator, self).__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)

        self.n_down = n_down

        self.state_fc = nn.Sequential(
            nn.Linear(self.state_dim,32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Linear(512,2048),
        )

        self.upnet = nn.Sequential(
            Up(512,512,norm_layer,use_bias),
            Up(512, 512, norm_layer, use_bias),
            Up(512,256,norm_layer,use_bias),
            Up(256, 256, norm_layer, use_bias),
            Up(256,128,norm_layer,use_bias),
            Up(128,64,norm_layer,use_bias),
            Up(64,32,norm_layer,use_bias)
        )

        self.outc = Outconv(ngf, 3)


    def forward(self, state):
        feature = self.state_fc(state).view(-1,512,2,2)

        grid = self.outc(self.upnet(feature))
        return grid


def img2state(opt=None, output_nc=3, ngf=32, n_down=6, norm='batch',
             use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_id='cuda:0'):
    norm_layer = get_norm_layer(norm_type=norm)
    input_nc = opt.stack_n*3
    net = StateGenerator(input_nc, output_nc, ngf, n_down, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, opt = opt)

    return init_net(net, init_type, init_gain, gpu_id)


class StateGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 n_down=6, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=9,
                 padding_type='reflect',opt=None):
        assert(n_blocks >= 0)
        super(StateGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)

        self.n_down = n_down
        self.opt = opt
        self.state_dim = opt.state_dim

        self.state_fc = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,self.state_dim),
        )

        self.downnet = nn.Sequential(
            Down(32,64,norm_layer,use_bias),
            Down(64,128,norm_layer,use_bias),
            Down(128,256,norm_layer,use_bias),
            Down(256, 256, norm_layer, use_bias),
            Down(256,512,norm_layer,use_bias),
            Down(512, 512, norm_layer, use_bias),
            Down(512,1024,norm_layer,use_bias)
        )

        self.pool = nn.AvgPool2d(2)


    def forward(self, img):
        img = self.inc(img)
        feature = self.downnet(img)
        feature = self.pool(feature).squeeze(-1).squeeze(-1)
        state = self.state_fc(feature)

        return state

def stateDmodel(opt=None):
    init_type='normal'
    init_gain=0.02
    gpu_id='cuda:0'
    net = StateD(opt=opt)
    return init_net(net, init_type, init_gain, gpu_id)


class StateD(nn.Module):
    def __init__(self,opt):
        super(StateD,self).__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.state_fc = nn.Sequential(
            nn.Linear(self.state_dim,32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )

    def forward(self, state):
        out = self.state_fc(state)
        return out


def imgDmodel(opt=None):
    init_type = 'normal'
    init_gain = 0.02
    gpu_id = 'cuda:0'

    net = ImgD()
    return init_net(net, init_type, init_gain, gpu_id)

class ImgD(nn.Module):
    def __init__(self):
        super(ImgD,self).__init__()
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

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)




if __name__ == '__main__':
    img = torch.rand(8, 3, 256, 256).float().cuda()
    state = torch.rand(8,3).float().cuda()

    Gx = img2state().cuda()
    out = Gx(img)
    print(out.shape)

    Gy = state2img().cuda()
    out = Gy(state).cuda()
    print(out.shape)

    Dx = imgDmodel().cuda()
    out = Dx(img)
    print(out.shape)

    Dy = stateDmodel().cuda()
    out = Dy(state)
    print(out.shape)
