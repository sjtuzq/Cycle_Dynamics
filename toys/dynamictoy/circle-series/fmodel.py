
import torch
import torch.nn as nn

from data import Fdata

class Fmodel(nn.Module):
    def __init__(self):
        super(Fmodel,self).__init__()
        self.statefc = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,32)
        )
        self.actionfc = nn.Sequential(
            nn.Linear(1, 16),
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
            nn.Linear(16,2)
        )
        self.load_state_dict(torch.load('./pred.pth'))

    def forward(self, state,action):
        state_feature = self.statefc(state)
        action_feature = self.actionfc(action.unsqueeze(1))
        feature = torch.cat((state_feature,action_feature),1)
        return self.predfc(feature)


if __name__ == '__main__':
    model = Fmodel().cuda()
    dataset = Fdata.get_loader()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        for i,item in enumerate(dataset):
            state,action,result = item
            state = state.float().cuda()
            action = action.float().cuda()
            result = result.float().cuda()

            out = model(state,action)
            loss = loss_fn(out,result)
            print(loss.item())

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

    # torch.save(model.state_dict(),'./pred.pth')