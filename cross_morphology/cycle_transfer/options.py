

import os
import argparse

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root' ,default='../../../logs/cross_morphology',type=str)
    parser.add_argument('--exp_id' ,default=10,type=int)
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument('--data_type1', type=str, default='base', help='data type')
    parser.add_argument('--data_type2', type=str, default='3leg', help='data type')
    parser.add_argument('--data_id1', type=int, default=1, help='data id')
    parser.add_argument('--data_id2', type=int, default=1, help='data id')

    parser.add_argument('--state_dim1' ,default=17 ,type=int)
    parser.add_argument('--action_dim1' ,default=6 ,type=int)

    parser.add_argument('--state_dim2', default=23, type=int)
    parser.add_argument('--action_dim2', default=9, type=int)

    parser.add_argument('--cut_state1', default=0, type=int)
    parser.add_argument('--cut_state2', default=0, type=int)


    parser.add_argument('--episode_n', default=200, type=int)
    parser.add_argument('--pair_n' ,default=7000 ,type=int)
    parser.add_argument('--display_gap' ,default=1000 ,type=int)
    parser.add_argument('--eval_gap' ,default=1000 ,type=int)
    parser.add_argument('--eval_n' ,default=3,type=int)

    parser.add_argument('--loss' ,default='l1' ,type=str)
    parser.add_argument('--istrain' ,default=True,type=bool)
    parser.add_argument('--pretrain_f' ,default=False,type=bool)

    parser.add_argument('--lr_Gx', default=1e-4, type=float)
    parser.add_argument('--lr_Ax', default=0., type=float)
    parser.add_argument('--lambda_G0', default=10., type=float)
    parser.add_argument('--lambda_G1', default=10., type=float)
    parser.add_argument('--lambda_GactA', default=10., type=float)
    parser.add_argument('--lambda_GactB', default=10., type=float)
    parser.add_argument('--lambda_Gcyc', default=30., type=float)
    parser.add_argument('--lambda_F', default=100., type=float)
    parser.add_argument('--init_start', default=True, type=bool)

    args = parser.parse_args()

    return args

