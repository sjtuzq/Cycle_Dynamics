
import torch
import argparse
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.stack_n = opt.stack_n
        self.backbone = resnet18(pretrained=True)
        self.fc = nn.Sequential(
            nn.Linear(1000*self.stack_n,64),
            nn.ReLU(),
            nn.Linear(64,self.state_dim),
            # nn.Tanh()
        )
        self.max = np.array([3,1.1,1.1,3,3,2,2.5,2.5])
        self.max = torch.tensor(self.max).float()

    def forward(self, obs):
        if self.stack_n>1:
            B,T,C,W,H = obs.shape
            feat = self.backbone(obs.view(-1,C,W,H))
            feat = feat.view(B,-1)
            pred = self.fc(feat)
            # pred = pred*self.max.cuda(device=pred.device)
            return pred.clamp(max=2.5,min=-2.5)
        else:
            feat = self.backbone(obs.squeeze(1))
            pred = self.fc(feat)
            # pred = pred * self.max.cuda(device=pred.device)
            return pred



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--state_dim', type=int, default=6, help='state dimension')
    parser.add_argument('--action_dim', type=int, default=4, help='action dimension')
    parser.add_argument('--stack_n', type=int, default=1, help='action dimension')

    opt = parser.parse_args()
    model = PixelEncoder(opt).cuda()

    obs = torch.rand(12,3,256,256).float().cuda()
    out = model(obs)

    print(obs.shape,out.shape)