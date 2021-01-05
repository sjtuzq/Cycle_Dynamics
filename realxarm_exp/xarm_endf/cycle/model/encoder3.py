
import torch
import torch.nn as nn
OUT_DIM = {1: 41, 2: 39, 3: 37, 4: 35, 6: 31}

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, opt=None,num_layers=3, num_filters=32):
        super().__init__()
        self.opt = opt
        self.input_c = opt.stack_n*3
        self.feature_dim = opt.state_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.input_c, num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=2))

        out_dim = OUT_DIM[num_layers]
        # self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.fc = nn.Linear(30752, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.out_fc = nn.Sequential(
            nn.Linear(30752,64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64,self.feature_dim)
        )

    def forward_conv(self, obs):
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs):
        h = self.forward_conv(obs)
        return self.out_fc(h)



if __name__ == '__main__':
    model = PixelEncoder().cuda()

    obs = torch.rand(1,9,84,84).float().cuda()
    out = model(obs)