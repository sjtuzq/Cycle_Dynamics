

import os
import torch
import argparse
from torchvision import transforms

from utils import init_logs
from neuraldata import CDFdata
from cyclechain import CycleGANModel


def train(opt):
    model = CycleGANModel(opt)
    dataset = CDFdata.get_loader(opt)
    img_logs,weight_logs = init_logs(opt)

    for epoch_id in range(opt.epoch_size):
        for batch_id, data in enumerate(dataset):
            model.set_input(data)
            model.optimize_parameters()

            if batch_id % opt.display_gap == 0:
                errors = model.get_current_errors()
                display = '===> Epoch[{}]({}/{})'.format(epoch_id, batch_id, len(dataset))
                for key, value in errors.items():
                    display += '{}:{:.4f}  '.format(key, value)
                print(display)

                path = os.path.join(img_logs, 'imgA_{}_{}.jpg'.format(epoch_id, batch_id))
                model.visual(path)
                model.save(weight_logs)


def eval(opt):
    model = CycleGANModel(opt)
    dataset = CDFdata.get_loader(opt)
    img_logs,weight_logs = init_logs(opt)
    model.load(weight_logs)

    for batch_id, data in enumerate(dataset):
        print('===> Epoch({}/{})'.format(batch_id, len(dataset)))
        model.set_input(data)
        model.test()
        path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id))
        model.visual(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--epoch_size', type=int, default=50, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--exp_id', type=int, default=3, help='experiment id')
    parser.add_argument('--test_id', type=int, default=19, help='dataset test id')
    parser.add_argument('--dynamic',type=bool,default=False,help='whether to use dynamic model')
    parser.add_argument('--clockwise', type=bool, default=False, help='dynamic direction')
    parser.add_argument('--domain',type=str,default='circle',help='the distribution of points')

    parser.add_argument('--loss', type=str, default='l1', help='loss function')
    parser.add_argument('--data_root', type=str, default='../../../logs', help='logs root path')
    parser.add_argument('--display_gap', type=int, default=25, help='training output frequency')

    parser.add_argument('--lambda_A', type=float, default=0., help='coefficient of lambdaA')
    parser.add_argument('--lambda_B', type=float, default=100., help='coefficient of lambdaB')
    parser.add_argument('--lambda_F', type=float, default=0., help='coefficient of lambdaF')

    opt = parser.parse_args()
    print(opt)
    train(opt)
    opt.istrain = False
    with torch.no_grad():
        eval(opt)
