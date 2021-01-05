

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
    # weight_logs = weight_logs.replace('exp_base_1_base_1','exp_base_1_base_1_bak30')
    # weight_logs = weight_logs.replace('weights','weights_bak2')
    print(weight_logs)
    model.load(weight_logs)
    # model.img_policy.online_test(model.netG_B,5)

    # return 0

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

    #
    # display ='average loss: '
    # for key, value in ave_loss.items():
    #     display += '{}:{:.4f}  '.format(key, value/len(dataset))
    # print(display)


if __name__ == '__main__':
    opt = get_options()


    # *****************************
    #           Reacher
    # *****************************
    opt.env = "Reacher-v2"
    opt.pretrain_f = False
    opt.data_id1 = 4
    opt.data_id2 = 4
    opt.G_lr = 1e-4
    opt.state_dim = 6
    opt.lambda_F = 200
    opt.lambda_G0 = 1
    opt.lambda_G1 = 1
    opt.stack_n = 1
    opt.f_epoch = 2
    opt.use_mask = True
    opt.mask = [3,3,0.4,0.4,2,2]
    opt.display_gap = 100
    train(opt)
    opt.istrain = False
    with torch.no_grad():
        eval(opt)

    # opt.env = "Reacher-v2"
    # opt.istrain = False
    # with torch.no_grad():
    #     opt.data_id1 = 0
    #     opt.data_id2 = 0
    #     eval(opt)






