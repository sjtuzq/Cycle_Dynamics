
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Inversemodel(nn.Module):
    def __init__(self,opt):
        super(Inversemodel,self).__init__()
        self.opt = opt
        self.state_dim = self.opt.state_dim
        self.action_dim = self.opt.action_dim
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim*2,256),
            nn.ReLU(),
            nn.Linear(256,self.action_dim),
            nn.Tanh()
        )
        self.max_action = 1.0

    def forward(self, now_state,nxt_state):
        state = torch.cat((now_state,nxt_state),1)
        action = self.fc(state)*self.max_action
        return action


class Forwardmodel(nn.Module):
    def __init__(self,opt):
        super(Forwardmodel,self).__init__()
        self.opt = opt
        self.action_dim = self.opt.action_dim
        self.state_dim = self.opt.state_dim
        self.state_fc = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU()
        )
        self.action_fc = nn.Sequential(
            nn.Linear(self.action_dim, 128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim)
        )

    def forward(self, state, action):
        action = self.action_fc(action)
        state = self.state_fc(state)
        state = torch.cat((action, state), 1)
        pred_state = self.fc(state)
        return pred_state


class Axmodel(nn.Module):
    def __init__(self,opt):
        super(Axmodel,self).__init__()
        self.opt = opt
        self.action_dim = self.opt.action_dim
        self.state_dim = self.opt.state_dim
        self.state_fc = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU()
        )
        self.action_fc = nn.Sequential(
            nn.Linear(self.action_dim, 128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
        )
        self.max_action = 1.0

    def forward(self, state, action):
        action = self.action_fc(action)
        state = self.state_fc(state)
        state = torch.cat((action, state), 1)
        action = self.fc(state)*self.max_action
        return action


class Dmodel(nn.Module):
    def __init__(self,opt):
        super(Dmodel,self).__init__()
        self.opt = opt
        self.action_dim = self.opt.action_dim
        self.state_dim = self.opt.state_dim
        self.state_fc = nn.Sequential(
            nn.Linear(self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, state):
        return self.state_fc(state)

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
