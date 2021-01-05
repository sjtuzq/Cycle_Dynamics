

import os
import torch
import argparse
from torchvision import transforms

from utils.utils import init_logs
from data.gymdata import RobotStackFdata
from data.gymdata import RobotStackdata as Robotdata
from model.cycle import CycleGANModel
from options import get_options


def train(opt):
    img_logs, weight_logs,tensor_writer = init_logs(opt)
    model = CycleGANModel(opt)
    dataset = Robotdata.get_loader(opt)
    fdataset = RobotStackFdata.get_loader(opt)
    model.train_forward_state(fdataset,opt.pretrain_f)
    model.parallel_init(device_ids=opt.device_ids)

    model.img_policy.online_test(model.netG_B, 1)

    niter = 0
    for epoch_id in range(opt.epoch_size):
        for batch_id, data in enumerate(dataset):
            model.set_input(data)
            model.optimize_parameters()

            niter += 1
            if (batch_id) % opt.display_gap == 0:
                errors = model.get_current_errors()
                display = '===> Epoch[{}]({}/{})'.format(epoch_id, batch_id, len(dataset))
                for key, value in errors.items():
                    display += '{}:{:.4f}  '.format(key, value)
                    tensor_writer.add_scalar(key, value, niter)
                print(display)

            if (batch_id) % opt.save_weight_gap == 0:
                path = os.path.join(img_logs, 'imgA_{}_{}.jpg'.format(epoch_id, batch_id+1))
                model.visual(path)
                model.save(weight_logs.replace('weights','weights_{}'.format(niter)))

                model.img_policy.online_test(model.netG_B,1)

def eval(opt):
    img_logs, weight_logs,tensor_writer = init_logs(opt)
    model = CycleGANModel(opt)
    model.parallel_init(device_ids=opt.device_ids)
    print(weight_logs)
    model.load(weight_logs)
    # model.img_policy.online_test(model.netG_B,5)

    dataset = Robotdata.get_loader(opt)
    ave_loss = {}
    for batch_id, data in enumerate(dataset):
        model.set_input(data)
        model.test()

        errors = model.get_current_errors()
        display = '===> Batch({}/{})'.format(batch_id, len(dataset))
        for key, value in errors.items():
            display += '{}:{:.4f}  '.format(key, value)
            try:
                ave_loss[key] = ave_loss[key] + value
            except:
                ave_loss[key] = value
        print(display)

        if (batch_id + 1) % opt.display_gap == 0:
            path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id + 1))
            model.visual(path)


if __name__ == '__main__':
    opt = get_options()

    # # *****************************
    # #        halfcheetah
    # # *****************************
    # opt.data_id1 = 1
    # opt.data_id2 = 1
    # opt.lambda_F = 200
    # opt.lambda_G0 = 10
    # opt.lambda_G1 = 10
    # # opt.state_dim = 8
    # opt.pretrain_f = False

    train(opt)
    opt.istrain = False
    with torch.no_grad():
        eval(opt)







