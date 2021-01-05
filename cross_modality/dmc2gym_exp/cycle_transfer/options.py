

import os
import argparse

def get_options():
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--istrain', type=bool, default=True, help='whether training or test')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--device_ids', type=list, default=[0,1,2,3], help='batch size')
    parser.add_argument('--exp_id', type=int, default=5, help='experiment id')
    parser.add_argument('--test_id1', type=int, default=1, help='dataset test id1')
    parser.add_argument('--test_id2', type=int, default=1, help='dataset test id2')

    parser.add_argument('--clip_range', type=int, default=5, help='action dimension')
    parser.add_argument('--stack_n', type=int, default=3, help='action dimension')
    parser.add_argument('--img_size', type=int, default=256, help='action dimension')
    parser.add_argument('--pretrain_f', type=bool, default=False, help='whether pretrained forward model')
    parser.add_argument('--f_epoch', type=int, default=2, help='whether pretrained forward model')
    parser.add_argument('--action_fix',type=bool,default=True,help='which action model to choose')
    parser.add_argument('--loss', type=str, default='l1', help='loss function')

    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--domain_name", default="cheetah")
    parser.add_argument("--task_name", default="run")
    parser.add_argument("--frame_skip", default=2)
    parser.add_argument("--force", type=bool, default=False)
    parser.add_argument("--log_root", default="../../../../../cross_modality")
    parser.add_argument('--episode_n', type=int, default=100, help='episode number')
    parser.add_argument('--state_dim', type=int, default=0, help='state dim')
    parser.add_argument('--action_dim', type=int, default=0, help='action dim')
    parser.add_argument('--eval_n', type=int, default=100, help='evaluation episode number')
    parser.add_argument('--epoch_n', type=int, default=30, help='training epoch number')
    parser.add_argument('--norm', type=bool, default=True, help='training epoch number')
    parser.add_argument('--use_mask', type=bool, default=False, help='training epoch number')
    parser.add_argument('--mask', type=list, default=[1,1,0,0,1,1], help='training epoch number')


    parser.add_argument('--data_type1', type=str, default='base', help='data type')
    parser.add_argument('--data_type2', type=str, default='base', help='data type')
    parser.add_argument('--data_id1', type=int, default=0, help='data id')
    parser.add_argument('--data_id2', type=int, default=0, help='data id')
    parser.add_argument('--display_gap', type=int, default=50, help = 'training output frequency')
    parser.add_argument('--save_weight_gap', type=int, default=300, help = 'training output frequency')

    parser.add_argument('--lambda_F', type=float, default=500., help='coefficient of lambdaF')
    parser.add_argument('--lambda_G0', type=float, default=30, help='coefficient of lambdaG0')
    parser.add_argument('--lambda_G1', type=float, default=30, help='coefficient of lambdaG1')
    parser.add_argument('--lambda_G2', type=float, default=0., help='coefficient of lambdaG2')
    parser.add_argument('--lambda_C', type=float, default=100., help='coefficient of lambdaC')
    parser.add_argument('--lambda_AC', type=float, default=100., help='coefficient of lambdaAC')
    parser.add_argument('--lambda_R', type=float, default=1000., help='coefficient of lambdaR')

    parser.add_argument('--F_lr', type=float, default=0, help='model F learning rate')
    parser.add_argument('--G_lr', type=float, default=1e-4, help='model G learning rate')
    parser.add_argument('--A_lr', type=float, default=0, help='action model learning rate')

    opt = parser.parse_args()
    print(opt)

    return opt