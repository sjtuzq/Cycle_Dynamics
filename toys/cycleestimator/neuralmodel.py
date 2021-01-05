

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
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
            # nn.ReLU(True)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(in_ch, out_ch,
            #           kernel_size=3, stride=1,
            #           padding=1, bias=use_bias),
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            norm_layer(out_ch),
            # nn.ReLU(True)
            nn.Tanh()
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


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net


def state2img(input_nc=3, output_nc=3, ngf=32, n_down=6, norm='batch',
             use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_id='cuda:0'):
    norm_layer = get_norm_layer(norm_type=norm)

    net = ImgGenerator(input_nc, output_nc, ngf, n_down, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    return init_net(net, init_type, init_gain, gpu_id)


class ImgGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 n_down=6, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ImgGenerator, self).__init__()
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
            nn.Linear(3,32),
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


def img2state(input_nc=3, output_nc=3, ngf=32, n_down=6, norm='batch',
             use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_id='cuda:0'):
    norm_layer = get_norm_layer(norm_type=norm)

    net = StateGenerator(input_nc, output_nc, ngf, n_down, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    return init_net(net, init_type, init_gain, gpu_id)


class StateGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 n_down=6, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
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

        self.state_fc = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
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

def stateDmodel(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = StateD()

    return init_net(net, init_type, init_gain, gpu_id)


class StateD(nn.Module):
    def __init__(self):
        super(StateD,self).__init__()
        self.state_fc = nn.Sequential(
            nn.Linear(3,32),
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



def imgDmodel(input_nc=3, ndf=64, netD='basic',
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = DModel()

    return init_net(net, init_type, init_gain, gpu_id)

class DModel(nn.Module):
    def __init__(self):
        super(DModel,self).__init__()
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
