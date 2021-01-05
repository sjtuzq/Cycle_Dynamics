#!/usr/bin/env bash
source ~/.virtualenvs/pytorch_env/bin/activate

# mpirun -np 1 python -u train.py --env-name='FetchReach-v1' --n-cycles=10 --n-epochs=10 2>&1 | tee reach.log

# python demo.py --env-name=FetchReach-v1