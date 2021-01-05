
import torch
import argparse
import torch.nn as nn

from neuraldata import CDFdata

class Fmodel(nn.Module):
    def __init__(self):
        super(Fmodel,self).__init__()
        self.statefc = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,32)
        )
        self.actionfc = nn.Sequential(
            nn.Linear(3, 16),
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
            nn.Linear(16,3)
        )

    def forward(self, state,action):
        state_feature = self.statefc(state)
        action_feature = self.actionfc(action)
        feature = torch.cat((state_feature,action_feature),1)
        return self.predfc(feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--exp_id', type=int, default=16, help='experiment id')
    parser.add_argument('--test_id', type=int, default=19, help='dataset test id')
    parser.add_argument('--dynamic',type=bool,default=True,help='whether to use dynamic model')
    parser.add_argument('--clockwise', type=bool, default=False, help='dynamic direction')
    parser.add_argument('--domain',type=str,default='circle',help='the distribution of points')

    parser.add_argument('--loss', type=str, default='l2', help='loss function')
    parser.add_argument('--imgA_id', type=int, default=10, help='datasetA from view1')
    parser.add_argument('--imgB_id',type=int,default=13,help='datasetB from view2')
    parser.add_argument('--data_root', type=str, default='../../../logs', help='logs root path')

    opt = parser.parse_args()

    model = Fmodel().cuda()
    dataset = CDFdata.get_loader(opt)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.L1Loss()

    for epoch in range(100):
        for i,item in enumerate(dataset):
            state,action,result = item[1]
            state = state.float().cuda()
            action = action.float().cuda()
            result = result.float().cuda()

            out = model(state,action)
            loss = loss_fn(out,result)
            # loss = loss_fn(result*0.5,(state*0.5+action*0.05))
            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(),'./pred.pth')