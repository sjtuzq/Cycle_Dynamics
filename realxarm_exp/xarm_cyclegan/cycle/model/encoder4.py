
import torch
import argparse
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.stack_n = opt.stack_n
        self.backbone1 = resnet18(pretrained=True)
        # self.backbone2 = resnet18(pretrained=True)
        self.fc1 = nn.Sequential(
            nn.Linear(1000*self.stack_n,64),
            nn.ReLU(),
            nn.Linear(64, self.state_dim),
            # nn.Sigmoid()
        )

    def forward(self, obs):
        feat1 = self.backbone1(obs)
        # feat2 = self.backbone2(obs)
        det1 = self.fc1(feat1)
        # det2 = self.fc2(feat2)
        # return torch.cat((det1,det2),1)
        return det1



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