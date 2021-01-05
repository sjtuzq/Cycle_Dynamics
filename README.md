# Cycle-Dynamics

This repository contains the official implementation for Cycle-Dynamics introduced in the following paper:

[**Learning Cross-Domain Correspondence for Control with Dynamics Cycle-Consistency**](https://arxiv.org/abs/2012.09811)

[Qiang Zhang](http://people.csail.mit.edu/qiangz/), [Tete Xiao](http://tetexiao.com/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Lerrel Pinto](https://cs.nyu.edu/~lp91/), [Xiaolong Wang](https://xiaolonw.github.io/)

The project page with video is at https://sjtuzq.github.io/cycle_dynamics.html.

## Citation
If you find our work useful in your research, please cite:
```
@article{zhang2020learning,
  title={Learning Cross-Domain Correspondence for Control with Dynamics Cycle-Consistency},
  author={Zhang, Qiang and Xiao, Tete and Efros, Alexei A and Pinto, Lerrel and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2012.09811},
  year={2020}
}
```

## Environment
- Python 3.6
- PyTorch 1.0 or higher, with NVIDIA CUDA Support
- Other required python packages are specified by `requirements.txt`.


## Prerequisites
- Install Mujoco and other required packages. 
- Clone this repository.
- Create the log folder './logs/cross_physics', './logs/cross_modality' and './logs/cross_morphology'.


## Reproducing Cross-physics Experiment
**1. Train the policy network, and collect the source domain data.** 
Or you can download the pretrained policy weight [here](https://drive.google.com/file/d/1oh_IxR0SxXyJ2WE5fGZN7Wzg_5r-yFDt/view?usp=sharing).
Then place it in './logs/cross_physics/', unzip this file and collect the source domian data.
```
cd cross_physics/base_train_test/td3_solver
python train.py --env 'HalfCheetah-v2'
cd ../../cycle_transfer
python collect_data.py --data_type 'base' --data_id 1
```

**2.Modify the xml file to change the environment, then collect data**
```
cd cross_physics/base_train_test/mujoco_xml
python modify_xml.py --env 'half_cheetah' --phy 'arma3'
cd ../../cycle_transfer
python collect_data.py --data_type 'arma3' --data_id 1
```

**3.Train the model to learn the correspondence**
```
cd cross_physics/cycle_transfer
python forwardexp.py --data_type1 'base' --data_id1 1 --data_type2 'arma3' --data_id2 1
```

## Reproducing Cross-modality Experiment
**1. Train the policy network and collect the source domain data.**
Or you can download the pretrained policy weight [here](https://drive.google.com/file/d/1oh_IxR0SxXyJ2WE5fGZN7Wzg_5r-yFDt/view?usp=sharing).
Then place it in './logs/cross_modality/', unzip this file and collect the source domian data.
```
cd cross_modality/mujoco_exp/base_train_test/td3_solver
python train.py --env 'HalfCheetah-v2'
```

**2.Collect the unpaired data for both domains.**
```
cd cross_modality/mujoco_exp/cycle_transfer/
python collect_data.py --env 'HalfCheetah-v2' --data_type 'base' --data_id 1
```

**3.Train the model to learn the correspondence.**
```
cd cross_modality/mujoco_exp/cycle_transfer/
python cycleexp.py --data_type1 'base' --data_id1 1 --data_type2 'base' --data_id2 1
```

## Reproducing Cross-morphology Experiment
**1. Train the policy network and collect the source domain data.**
Or you can download the pretrained policy weight [here](https://drive.google.com/file/d/1oh_IxR0SxXyJ2WE5fGZN7Wzg_5r-yFDt/view?usp=sharing).
Then place it in './logs/cross_morphology/', unzip this file and collect the source domian data.
```
cd cross_morphology/base_train_test/td3_solver
python train.py --env 'HalfCheetah-v2'
cd ../../cycle_transfer
python collect_data.py --data_type 'base' --data_id 1
```

**2.Modify the xml file to change the agent morphology.**
```
cd cross_morphology/base_train_test/mujoco_xml
python modify_xml.py --env 'half_cheetah' --phy '3leg'
cd ../../cycle_transfer
python collect_data.py --data_type '3leg' --data_id 1
```

**3.Train the model to learn the correspondence.**
```
cd cross_morphology/cycle_transfer
python alignexp.py --data_type1 'base' --data_id1 1 --data_type2 '3leg' --data_id2 1
```

## Reproducing Real Xarm Robot Experiment
Download the dataset here

To train and evaluate the model:
```
cd realxarm_exp/xarm_pose
python main.py
```

## Toy Experiment
The code for toy experiment is shown in the path './toys', which shows how we explore step by step to get the final framework.
This part is not cleaned up but you can still have a reference if you are interested in.