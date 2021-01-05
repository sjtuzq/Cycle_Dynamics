#!/usr/bin/env bash

python train.py --env HalfCheetah-v2
python train.py --env Hopper-v2
python train.py --env Walker2d-v2
python train.py --env Ant-v2
python train.py --env Humanoid-v2
python train.py --env Swimmer-v2


python test.py --env HalfCheetah-v2
python test.py --env Hopper-v2
python test.py --env Walker2d-v2
python test.py --env Ant-v2
python test.py --env Humanoid-v2
python test.py --env Swimmer-v2