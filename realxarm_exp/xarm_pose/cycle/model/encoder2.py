
import torch
import torch.nn as nn


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


class PixelEncoder(nn.Module):
    def __init__(self, opt=None,num_layers=2, num_filters=32):
        super().__init__()
        self.opt = opt
        self.input_c = opt.stack_n*3
        self.feature_dim = opt.state_dim
        self.num_layers = num_layers

        self.opt = opt
        feature_dim = 512

        self.encoder = nn.Sequential(
            Down(self.input_c,32),
            Down(32,64),
            Down(64,128),
            Down(128,256),
            Down(256,256),
            Down(256,256),
            Down(256,256),
            Down(256,feature_dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(feature_dim,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.feature_dim)
        )

    def forward(self, data):
        feature = self.encoder(data)
        pred = self.fc(feature.squeeze())
        return pred
