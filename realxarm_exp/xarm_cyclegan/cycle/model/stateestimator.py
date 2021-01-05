

import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import sys
sys.path.append('../')

from model.dgmodel import img2state
from data.gymdata import Robotdata,RobotStackdata
from model.encoder import PixelEncoder
# from model.encoder2 import PixelEncoder
# from model.encoder3 import PixelEncoder

def train_imgf(opt):
    model = img2state(opt).cuda()
    dataset = Robotdata.get_loader(opt)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        for i,item in enumerate(dataset):
            state,action,result = item[0]
            state = state.float().cuda()
            action = action.float().cuda()
            result = result.float().cuda()

            out = model(state,action)
            loss = loss_fn(out,result)*100
            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(),'./imgpred.pth')

        # generate the batch image
        merge_img = []
        for (before, after, pred) in zip(state, result, out):
            img = torch.cat([before, after, pred], 1)
            merge_img.append(img)
        merge_img = torch.cat(merge_img, 2).cpu()
        merge_img = (merge_img + 1) / 2
        img = transforms.ToPILImage()(merge_img)
        # img = transforms.Resize((512, 640))(img)
        img.save(os.path.join('./tmp/imgpred', 'img_{}.jpg'.format(epoch)))


def train_img2state(opt):
    # model = img2state(opt).cuda()
    # weight_path = os.path.join(opt.data_root, 'data_{}/img2state.pth'.format(opt.test_id1))
    # model.load_state_dict(torch.load(weight_path))

    model = PixelEncoder(opt).cuda()
    model = nn.DataParallel(model,device_ids=[0,1,2,3])
    weight_path = os.path.join(opt.data_root, 'data_{}/img2state_large.pth'.format(opt.test_id1))
    try:
        model.load_state_dict(torch.load(weight_path))
        print('continue training!')
    except:
        print('training from scratch!')

    dataset = RobotStackdata.get_loader(opt)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    loss_fn = nn.L1Loss()

    for epoch in range(opt.epoch_size):
        epoch_loss = 0
        for i, item in enumerate(tqdm(dataset)):
            state, action, result = item[1]
            input_Bt0 = item[0][0]
            input_Bt1 = item[0][2]
            action = item[0][1]
            gt0 = item[2][0].float().cuda()
            gt1 = item[2][1].float().cuda()

            img = input_Bt0.float().cuda()
            gt = gt0.float().cuda()

            out = model(img)
            loss = loss_fn(out, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print('epoch:{} loss:{:.7f}'.format(epoch, epoch_loss / len(dataset)))
        weight_path = os.path.join(opt.data_root,'data_{}/img2state_large.pth'.format(opt.test_id1))
        torch.save(model.state_dict(),weight_path)


def eval_img2state(opt):
    model = PixelEncoder(opt).cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
    weight_path = os.path.join(opt.data_root, 'data_{}/img2state_large.pth'.format(opt.test_id1))
    model.load_state_dict(torch.load(weight_path))

    dataset = RobotStackdata.get_loader(opt)
    loss_fn = nn.L1Loss()

    epoch_loss = 0
    origin,recover = [],[]
    for i, item in enumerate(dataset):
        state, action, result = item[1]
        input_Bt0 = item[0][0]
        input_Bt1 = item[0][2]
        action = item[0][1]
        gt0 = item[2][0].float().cuda()
        gt1 = item[2][1].float().cuda()

        img = input_Bt0.float().cuda()
        gt = gt0.float().cuda()

        out = model(img)
        loss = loss_fn(out, gt)
        epoch_loss += loss.item()

        print(i,epoch_loss/(i+1))

        origin.append(gt.cpu().data.numpy())
        recover.append(out.cpu().data.numpy())

        if i>100:
            break

    print('epoch:{} loss:{:.7f}'.format(0, epoch_loss / len(dataset)))

    origin = np.vstack(origin)
    recover = np.vstack(recover)

    np.save(os.path.join(opt.data_root,'data_{}/origin.npy'.format(opt.test_id1)),origin)
    np.save(os.path.join(opt.data_root,'data_{}/recover.npy'.format(opt.test_id1)),recover)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=14*4, help='batch size')
    parser.add_argument('--exp_id', type=int, default=3, help='experiment id')
    parser.add_argument('--test_id1', type=int, default=5, help='dataset test id')
    parser.add_argument('--test_id2', type=int, default=5, help='dataset test id')

    # fetch series: state_dim=25, action_dim=4
    # control series: state_dim=4, action_dim=1
    # walk series: state_dim=17, action_dim=6
    parser.add_argument('--state_dim', type=int, default=17, help='state dimension')
    parser.add_argument('--action_dim', type=int, default=6, help='action dimension')
    parser.add_argument('--stack_n', type=int, default=3, help='action dimension')
    parser.add_argument('--img_size', type=int, default=84, help='action dimension')

    parser.add_argument('--loss', type=str, default='l2', help='loss function')
    parser.add_argument('--imgA_id', type=int, default=1, help='datasetA from view1')
    parser.add_argument('--imgB_id',type=int,default=2,help='datasetB from view2')
    parser.add_argument('--data_root', type=str, default='../../../logs', help='logs root path')

    opt = parser.parse_args()
    train_img2state(opt)
    with torch.no_grad():
        eval_img2state(opt)
