

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

    best_reward = 0
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
                reward = model.img_policy.online_test(model.netG_B, 1)
                reward = reward.mean()
                if reward>best_reward:
                    best_reward = reward
                    model.save(weight_logs)
                print('best_reward:{}  cur_reward:{}'.format(best_reward,reward))


def eval(opt):
    img_logs, weight_logs,tensor_writer = init_logs(opt)
    model = CycleGANModel(opt)
    model.parallel_init(device_ids=opt.device_ids)
    weight_logs = weight_logs.replace('weights','weights_bak3')
    print(weight_logs)
    model.load(weight_logs)
    # model.netG_B.eval()
    model.img_policy.online_test(model.netG_B,5,img_logs)

    # dataset = Robotdata.get_loader(opt)
    # ave_loss = {}
    # for batch_id, data in enumerate(dataset):
    #     model.set_input(data)
    #     model.test()
    #
    #     errors = model.get_current_errors()
    #     display = '===> Batch({}/{})'.format(batch_id, len(dataset))
    #     for key, value in errors.items():
    #         display += '{}:{:.4f}  '.format(key, value)
    #         try:
    #             ave_loss[key] = ave_loss[key] + value
    #         except:
    #             ave_loss[key] = value
    #     print(display)
    #
    #     # if (batch_id + 1) % opt.display_gap == 0:
    #     #     path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id + 1))
    #     #     model.visual(path)
    #
    # display ='average loss: '
    # for key, value in ave_loss.items():
    #     display += '{}:{:.4f}  '.format(key, value/len(dataset))
    # print(display)


if __name__ == '__main__':
    opt = get_options()
    opt.lambda_G0 = 1
    opt.lambda_G1 = 1
    opt.lambda_G2 = 1
    opt.lambda_F = 500

    # opt.data_id1 = 1
    # opt.data_id2 = 1
    # opt.pretrain_f = False
    # train(opt)

    # opt.pretrain_f = True
    # opt.norm = True
    # opt.domain_name = 'finger'
    # opt.task_name = 'spin'
    # opt.data_id1 = 3
    # opt.data_id2 = 3
    # opt.state_dim = 6
    # opt.action_dim = 2
    # opt.lambda_F = 300
    # opt.frame_skip = 1
    # opt.use_mask = True
    # opt.mask = [3,3,0.5,0.5,1,1]
    # # train(opt)
    # opt.istrain = False
    # eval(opt)


    opt.pretrain_f = False
    opt.norm = True
    opt.domain_name = 'reacher'
    opt.task_name = 'hard'
    opt.data_id1 = 2
    opt.data_id2 = 2
    opt.state_dim = 2
    opt.action_dim = 2
    opt.lambda_F = 300
    opt.frame_skip = 1
    opt.f_epoch = 5
    # opt.use_mask = True
    # opt.mask = [3,3,0.2,0.2,2,2]
    train(opt)
    opt.istrain = False
    eval(opt)

    # opt.pretrain_f = False
    # opt.norm = True
    # opt.domain_name = 'manipulator'
    # opt.task_name = 'bring_ball'
    # opt.state_dim = 44
    # opt.action_dim = 5
    # train(opt)

    # opt.pretrain_f = True
    # opt.domain_name = 'pendulum'
    # opt.task_name = 'swingup'
    # # opt.lambda_G0 = 1
    # # opt.lambda_G1 = 0
    # # opt.lambda_G2 = 0
    # opt.data_id1 = 2
    # opt.data_id2 = 2
    # opt.state_dim = 2
    # opt.action_dim = 1
    # opt.lambda_F = 500
    # train(opt)







