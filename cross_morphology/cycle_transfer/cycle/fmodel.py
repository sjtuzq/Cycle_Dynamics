

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

class Fmodel(nn.Module):
    def __init__(self,opt):
        super(Fmodel,self).__init__()
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


class Fengine:
    def __init__(self,opt):
        self.fmodel = Fmodel(opt).cuda()
        self.opt = opt

    def train_statef(self,dataset):
        optimizer = torch.optim.Adam(self.fmodel.parameters(),lr=1e-4)
        loss_fn = nn.L1Loss()
        now,act,nxt = dataset
        now = now[:,:self.opt.state_dim]
        nxt = nxt[:,:self.opt.state_dim]
        batch_size = 32
        for epoch in range(50):
            epoch_loss,cmp_loss = 0,0
            for i in tqdm(range(int(now.shape[0]/batch_size))):
                start = i*batch_size
                end = start+batch_size
                state = torch.tensor(now[start:end]).float().cuda()
                action = torch.tensor(act[start:end]).float().cuda()
                result = torch.tensor(nxt[start:end]).float().cuda()

                out = self.fmodel(state, action)
                loss = loss_fn(out, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                cmp_loss += loss_fn(state,result).item()
            print('epoch:{} loss:{:.7f} cmp:{:.7f}'.format(epoch, epoch_loss / len(dataset),cmp_loss / len(dataset)))
            weight_path = './env/data/data{}/pred_weight.pth'.format(self.opt.data_id1)
            torch.save(self.fmodel.state_dict(),weight_path)

