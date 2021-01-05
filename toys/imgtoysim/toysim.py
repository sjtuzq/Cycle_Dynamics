
"""
This toy expriment is about multi-view camera image transfer using cyclegan

"""
import os
import torch
import argparse
from torchvision import transforms

from simmodule.utils import init_logs
from simmodule.data import Mydata
from simmodule.cyclechain import CycleGANModel

def train(opt):
    model = CycleGANModel(opt)
    dataset = Mydata.get_loader(opt)
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
    dataset = Mydata.get_loader(opt)
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
    parser.add_argument('--epoch_size', type=int, default=20, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--exp_id', type=int, default=7, help='experiment id')
    parser.add_argument('--loss', type=str, default='l2', help='loss function')
    parser.add_argument('--imgA_id', type=int, default=10, help='datasetA from view1')
    parser.add_argument('--imgB_id',type=int,default=13,help='datasetB from view2')
    parser.add_argument('--lambda_AB', type=float, default=40., help='cycle-gan generator gamma')
    parser.add_argument('--data_root', type=str, default='../../logs', help='logs root path')
    parser.add_argument('--display_gap', type=int, default=100, help='training output frequency')
    opt = parser.parse_args()
    print(opt)
    train(opt)
    opt.istrain = False
    with torch.no_grad():
        eval(opt)
