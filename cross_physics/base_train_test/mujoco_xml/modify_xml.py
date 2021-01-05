

import os
import sys
import argparse

def modify(args):
    xml_path = os.path.join(args.local_path, '{}/{}_{}.xml'.format(args.env,args.env, args.phy))
    remote_path = os.path.join(sys.path[4],args.lib_path, '{}.xml'.format(args.env))
    xml_data = open(xml_path, 'r').read()
    print(xml_data)
    with open(remote_path,'w') as f:
        f.write(xml_data)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib_path',default='site-packages/gym/envs/mujoco/assets')
    parser.add_argument('--local_path',default='./assets')
    parser.add_argument('--env',default='half_cheetah')
    parser.add_argument('--phy',default='arma3')

    args = parser.parse_args()
    modify(args)







