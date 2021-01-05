

import os
import torch
import numpy as np
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

    # model.img_policy.online_test(model.netG_B, 1)

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

                # model.img_policy.online_test(model.netG_B,1)

def eval(opt):
    img_logs, weight_logs,tensor_writer = init_logs(opt)
    model = CycleGANModel(opt)
    model.parallel_init(device_ids=opt.device_ids)
    print(weight_logs)
    dataset = Robotdata.get_loader(opt)
    # fdataset = RobotStackFdata.get_loader(opt)
    # model.train_forward_state(fdataset, True)
    weight_list = list(filter(lambda x:'weights_' in x,os.listdir(weight_logs.replace('weights',''))))
    weight_list = sorted(weight_list,key=lambda x:int(x.split('_')[1]))

    for weight_path in weight_list:
        weight_id = int(weight_path.split('_')[1])
        weight_path = weight_logs.replace('weights',weight_path)
        print(weight_path)
        model.load(weight_path)

        # model.img_policy.online_test(model.netG_B,5)

        dataset = Robotdata.get_loader(opt)
        ave_loss = {}
        count_n = 10
        gt, pred = [], []
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
            gt.append(model.gt0.cpu().data.numpy())
            pred.append(model.fake_At0.cpu().data.numpy())

            if batch_id>=count_n-1:
                path = os.path.join(img_logs, 'imgA_{}.jpg'.format(weight_id))
                model.visual(path)
                break

        gt = np.vstack(gt)
        pred = np.vstack(pred)
        np.save(os.path.join(img_logs,'gt_{}.npy'.format(weight_id)),gt)
        np.save(os.path.join(img_logs,'pred_{}.npy'.format(weight_id)),pred)

        display ='average loss: '
        for key, value in ave_loss.items():
            display += '{}:{:.4f}  '.format(key, value/(count_n))
        print(display)


if __name__ == '__main__':
    opt = get_options()

    # *****************************
    #        halfcheetah
    # *****************************

    opt.data_id1 = 1
    opt.data_id2 = 1
    opt.lambda_F = 200
    opt.lambda_G0 = 0
    opt.lambda_G1 = 0
    # opt.state_dim = 8
    opt.pretrain_f = True
    # train(opt)
    opt.istrain = False
    with torch.no_grad():
        opt.data_id1 = 1
        opt.data_id2 = 1
        eval(opt)


    # *****************************
    #           hopper
    # *****************************
    # opt.env = "Hopper-v2"
    # opt.pretrain_f = True
    # opt.lambda_F = 0
    # opt.lambda_G0 = 10
    # opt.lambda_G1 = 0
    # train(opt)
    # opt.istrain = False
    # with torch.no_grad():
    #     eval(opt)

    # opt.env = "Hopper-v2"
    # opt.istrain = False
    # with torch.no_grad():
    #     opt.data_id1 = 0
    #     opt.data_id2 = 0
    #     eval(opt)


    # *****************************
    #           walker2d
    # *****************************
    # opt.env = "Walker2d-v2"
    # opt.pretrain_f = False
    # opt.lambda_F = 300
    # opt.lambda_G0 = 1
    # opt.lambda_G1 = 1
    # opt.use_mask = False
    # train(opt)
    # opt.istrain = False
    # with torch.no_grad():
    #     eval(opt)

    # opt.env = "Walker2d-v2"
    # opt.istrain = False
    # with torch.no_grad():
    #     opt.data_id1 = 0
    #     opt.data_id2 = 0
    #     eval(opt)

    # *****************************
    #           ant
    # *****************************
    # opt.env = "Ant-v2"
    # opt.pretrain_f = True
    # opt.lambda_F = 100
    # opt.lambda_G0 = 0
    # opt.lambda_G1 = 0
    # opt.state_dim = 27
    # train(opt)
    # opt.istrain = False
    # with torch.no_grad():
    #     eval(opt)

    # opt.env = "Ant-v2"
    # opt.istrain = False
    # with torch.no_grad():
    #     opt.data_id1 = 0
    #     opt.data_id2 = 0
    #     eval(opt)


    # *****************************
    #           swimmer
    # *****************************
    # opt.env = "Swimmer-v2"
    # opt.pretrain_f = False
    # opt.data_id1 = 10
    # opt.data_id2 = 10
    # # opt.G_lr = 1e-3
    # opt.lambda_F = 300
    # opt.lambda_G0 = 1
    # opt.lambda_G1 = 1
    # opt.state_dim = 3
    # opt.stack_n = 3
    # opt.use_mask = False
    # train(opt)
    # opt.istrain = False
    # with torch.no_grad():
    #     eval(opt)

    # opt.env = "Swimmer-v2"
    # opt.istrain = False
    # with torch.no_grad():
    #     opt.data_id1 = 0
    #     opt.data_id2 = 0
    #     eval(opt)


    # *****************************
    #           InvertedDoublePendulum
    # *****************************
    # opt.env = "InvertedDoublePendulum-v2"
    # opt.pretrain_f = True
    # opt.data_id1 = 0
    # opt.data_id2 = 0
    # # opt.G_lr = 1e-5
    # opt.state_dim = 8
    # opt.lambda_F = 100
    # opt.lambda_G0 = 1
    # opt.lambda_G1 = 1
    # opt.stack_n = 2
    # train(opt)
    # opt.istrain = False
    # with torch.no_grad():
    #     eval(opt)

    # opt.env = "InvertedDoublePendulum-v2"
    # opt.istrain = False
    # with torch.no_grad():
    #     opt.data_id1 = 0
    #     opt.data_id2 = 0
    #     eval(opt)

    # *****************************
    #           Reacher
    # *****************************
    # opt.env = "Reacher-v2"
    # opt.pretrain_f = True
    # opt.data_id1 = 1
    # opt.data_id2 = 1
    # opt.G_lr = 1e-4
    # opt.state_dim = 10
    # opt.lambda_F = 200
    # opt.lambda_G0 = 1
    # opt.lambda_G1 = 1
    # opt.stack_n = 1
    # opt.display_gap = 100
    # train(opt)
    # opt.istrain = False
    # with torch.no_grad():
    #     eval(opt)

    # opt.env = "Reacher-v2"
    # opt.istrain = False
    # with torch.no_grad():
    #     opt.data_id1 = 0
    #     opt.data_id2 = 0
    #     eval(opt)






