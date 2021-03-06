

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from neuraldata import Fdata
from neuralmodel import Flow


def train(args):
    dataset = Fdata.get_loader(opt=args)
    model = Flow().cuda()
    Optimizer = optim.Adam(model.parameters())
    recon_loss = nn.L1Loss()

    print('------------  is training  ------------------')
    niter, ave_l1, ave_l2, ave_lr, ave_aprn,ave_pose = 1, 0, 0, 0, 0, 0
    for epoch_id in range(args.epoch_size):
        for batch_id,item in enumerate(dataset):
            img = item[0].cuda().float()
            state = item[1].cuda().float()

            out = model(state)

            loss = recon_loss(out,img)

            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

            print(epoch_id,batch_id,loss.item())

        torch.save(model.state_dict(),'./tmp/model.pth')


def test(args):
    dataset = Fdata.get_loader(opt=args)
    model = Flow().cuda()
    model.load_state_dict(torch.load('./tmp/model.pth'))


    print('------------  is training  ------------------')
    niter, ave_l1, ave_l2, ave_lr, ave_aprn,ave_pose = 1, 0, 0, 0, 0, 0
    for batch_id,item in enumerate(dataset):
        img = item[0].cuda().float()
        state = item[1].cuda().float()
        out = model(state)

        merge_img = []
        for item in torch.cat((img,out), 2):
            merge_img.append(item)
        merge_img = torch.cat(merge_img, 2).cpu()

        merge_img = (merge_img+1)/2

        img = transforms.ToPILImage()(merge_img)
        img.save('./tmp/img_{}.jpg'.format(batch_id))

        if batch_id>40:
            break
        print(batch_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--epoch_size', type=int, default=50, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--test_id', type=int, default=19, help='dataset id')
    parser.add_argument('--exp_id', type=int, default=1, help='experiment id')
    parser.add_argument('--dynamic',type=bool,default=True,help='whether to use dynamic model')
    parser.add_argument('--loss', type=str, default='l2', help='loss function')
    parser.add_argument('--imgA_id', type=int, default=12, help='datasetA from view1')
    parser.add_argument('--imgB_id',type=int,default=13,help='datasetB from view2')
    parser.add_argument('--lambda_AB', type=float, default=20., help='generator loss gamma')
    parser.add_argument('--lambda_F', type=float, default=10., help='forward model loss gamma')
    parser.add_argument('--data_root', type=str, default='../../../logs', help='logs root path')
    parser.add_argument('--display_gap', type=int, default=100, help='training output frequency')
    opt = parser.parse_args()
    # train(opt)
    opt.istrain = False
    test(opt)

