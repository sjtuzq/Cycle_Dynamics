

import os
import torch
import argparse
import numpy as np
from torchvision import transforms

from cycle.utils import init_logs
from cycle.data.realdata import BoltData
from cycle.model.cycle import CycleGANModel

def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def train(opt):
    img_logs, weight_logs,tensor_writer = init_logs(opt)
    dataset = BoltData.get_loader(opt)
    model = CycleGANModel(opt)
    model.parallel_init([0,1,2,3])

    niter,best_loss = 0,100
    for epoch_id in range(opt.epoch_size):
        for batch_id, data in enumerate(dataset):
            model.set_input(data)
            try:
                model.optimize_parameters()
            except:
                continue

            if (batch_id) % opt.display_gap == 0:
                errors = model.get_current_errors()
                display = '\n===> Epoch[{}]({}/{})'.format(epoch_id, batch_id, len(dataset))
                for key, value in errors.items():
                    display += '{}:{:.4f}  '.format(key, value)
                    tensor_writer.add_scalar(key, value, niter)
                    niter += 1
                print(display)

                cur_loss = errors['L_t0']
                if cur_loss<best_loss:
                    best_loss = cur_loss
                    model.save(weight_logs)
                path = os.path.join(img_logs, 'img_batch_{}.jpg'.format(niter))
                model.visual(path)


def get_state(opt):
    img_logs, weight_logs,tensor_writer = init_logs(opt)
    dataset = BoltData.get_loader(opt)
    model = CycleGANModel(opt)
    model.parallel_init([0, 1, 2, 3])
    model.load(weight_logs)

    pred, gt = [], []
    for batch_id, data in enumerate(dataset):
        model.set_input(data)
        model.test()

        pred.append(model.fake_A)
        gt.append(model.gt0)

        print(batch_id)

        # if batch_id>20:
        #     break

    pred = torch.cat(pred, 0).cpu().data.numpy()
    gt = torch.cat(gt, 0).cpu().data.numpy()
    print(abs(pred-gt).mean(0))

    np.save(weight_logs.replace('weights', 'pred_z.npy'), pred)
    np.save(weight_logs.replace('weights', 'gt_z.npy'), gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--data_dir', type=str,
                        default='/home/xiaolonw/zqdata/workspace/cycle/reallogs/dataset/xarm/v2',
                        help='Data directory to visualize')

    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--exp_id', type=int, default=1, help='experiment id')
    parser.add_argument('--test_id1', type=int, default=4, help='dataset test id1')
    parser.add_argument('--test_id2', type=int, default=4, help='dataset test id2')

    parser.add_argument('--state_dim', type=int, default=3, help='state dimension')
    parser.add_argument('--action_dim', type=int, default=3, help='action dimension')
    parser.add_argument('--state_type', type=str, default='end', help='state representation')
    parser.add_argument('--clip_range', type=int, default=4, help='action dimension')
    parser.add_argument('--stack_n', type=int, default=1, help='action dimension')
    parser.add_argument('--img_size', type=int, default=256, help='action dimension')
    parser.add_argument('--pretrain_f', type=bool, default=False, help='whether pretrained forward model')
    parser.add_argument('--f_epoch', type=int, default=4, help='how many epochs to train froward model')
    parser.add_argument('--action_fix',type=bool,default=True,help='which action model to choose')
    parser.add_argument('--moving_stage',type=bool,default=True,help='which stage')

    parser.add_argument('--loss', type=str, default='l1', help='loss function')
    parser.add_argument('--imgA_id', type=int, default=7, help='datasetA from view1')
    parser.add_argument('--imgB_id',type=int,default=8,help='datasetB from view2')
    parser.add_argument('--data_root', type=str, default='../../../reallogs', help='logs root path')
    parser.add_argument('--display_gap', type=int, default=40, help = 'training output frequency')

    parser.add_argument('--lambda_A', type=float, default=10., help='coefficient of lambdaF')
    parser.add_argument('--lambda_B', type=float, default=10., help='coefficient of lambdaF')

    parser.add_argument('--F_lr', type=float, default=0, help='model F learning rate')
    parser.add_argument('--G_lr', type=float, default=1e-4, help='model G learning rate')
    parser.add_argument('--A_lr', type=float, default=0, help='action model learning rate')

    opt = parser.parse_args()

    if opt.state_type=='joint':
        opt.state_dim = 7
    print(opt)
    # train(opt)
    opt.istrain = False
    with torch.no_grad():
        get_state(opt)

