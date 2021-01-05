import numpy as np
import argparse
import os
import json
import pickle
import random
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch.utils.data as Data
from torchvision import transforms

def get_secs(val):
    val = val.split('.')
    v0 = int(val[0])
    v1 = int((val[1] + 7 * '0')[:7])
    return v0 + (1e-7) * float(v1)


class BoltData(Data.Dataset):
    def __init__(self,opt):
        self.opt = opt
        self.data_root = opt.data_dir
        self.trans = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        self.now_img,self.act_img,self.nxt_img,self.now_gt,self.nxt_gt = self.parse_real_data(self.data_root)
        self.now_state,self.act_state,self.nxt_state = self.parse_sim_data()
        self.sample_num2 = len(self.now_img)
        self.sample_num1 = self.now_state.shape[0]

    def parse_sim_data(self):
        sim_pair = 10000
        now_pos = np.random.rand(sim_pair,3)
        now_pos = now_pos*(self.now_gt.max(0)-self.now_gt.min(0))+self.now_gt.min(0)
        act = np.random.rand(sim_pair,3)
        act = act*(self.act_img.max(0)-self.act_img.min(0))+self.act_img.min(0)
        nxt_pos = now_pos+act
        nxt_pos = np.clip(nxt_pos,a_max=self.now_gt.max(0),a_min=self.now_gt.min(0))

        sim_n = now_pos.shape[0]
        assert (sim_n==act.shape[0])
        assert (sim_n==nxt_pos.shape[0])
        return now_pos,act,nxt_pos


    def parse_real_data(self,data_root):
        all_now,all_act,all_nxt = [],[],[]
        all_s1,all_s2 = [],[]
        data_list = ['2','3','4','5','6']
        data_list = ['3']
        data_list = ['1']
        for dir in os.listdir(data_root):
            if not dir in data_list:
                continue
            print(dir)
            try:
                now,act,nxt,s1,s2 = self.parse_one_scene(os.path.join(data_root,dir))
            except:
                continue
            all_now.extend(now)
            all_act.extend(act)
            all_nxt.extend(nxt)
            all_s1.append(s1)
            all_s2.append(s2)
        pair_n = len(all_now)
        assert (pair_n==len(all_act))
        assert (pair_n==len(all_nxt))
        all_act = np.vstack(all_act)
        all_s1 = np.vstack(all_s1)
        all_s2 = np.vstack(all_s2)

        # self.mean = all_s1.mean(0)
        # self.std = all_s1.std(0)
        # all_s1 = (all_s1-self.mean)
        # all_s2 = (all_s2-self.mean)
        # all_act = all_act/all_act.max(0)

        print('total data: {}'.format(pair_n))
        self.pair_n = pair_n
        return all_now,all_act,all_nxt,all_s1,all_s2

    def parse_one_scene(self,data_dir):
        image_dir = os.path.join(data_dir, 'images')
        data_dict = pickle.load(
                    open(os.path.join(data_dir,'data_dict.pkl'), 'rb'),
                    encoding='iso-8859-1')

        data_dict = [data_dict[i] for i in range(len(data_dict))]
        state_act = [np.concatenate((x['joint_obs'],x['end_eff_obs']),0) for x in data_dict]

        state_buffer = np.stack(state_act)[:,:7]
        end_pos = np.stack(state_act)[:,7:10]

        frame_buffer = sorted(os.listdir(image_dir),key=get_secs)
        frame_buffer = [os.path.join(image_dir,x) for x in frame_buffer]

        if self.opt.istrain:
            cut_id_list = [1, 2]
        else:
            cut_id_list = [1]
        now_img,nxt_img,now_state,nxt_state,action = [],[],[],[],[]
        for cut_id in cut_id_list:
            now_img.extend(frame_buffer[:-cut_id])
            nxt_img.extend(frame_buffer[cut_id:])

            # now_state.append(state_buffer[:-cut_id])
            # nxt_state.append(state_buffer[cut_id:])
            now_state.append(end_pos[:-cut_id])
            nxt_state.append(end_pos[cut_id:])
            action.append(end_pos[cut_id:] - end_pos[:-cut_id])

        now_state = np.vstack(now_state)
        nxt_state = np.vstack(nxt_state)
        action = np.vstack(action)

        pair_n = len(now_img)
        assert (pair_n==len(nxt_img))
        assert (pair_n==action.shape[0])
        assert (pair_n==now_state.shape[0])
        assert (pair_n==nxt_state.shape[0])

        print('loading {} pair data!'.format(pair_n))

        return now_img,action,nxt_img,now_state,nxt_state


    def get_state_sample(self,id=None):
        if id is None:
            id = random.sample(range(self.sample_num1), 1)[0]
        return self.now_state[id],self.act_state[id],self.nxt_state[id]

    def path2img(self,path):
        img = Image.open(path)
        img = transforms.ToTensor()(img)
        img = img[:, 60:400, 160:500]
        img = transforms.ToPILImage()(img)
        img = transforms.Resize((256,256))(img)
        img = transforms.ToTensor()(img)
        return img

    def get_img_sample(self,id=None):
        try:
            if id is None:
                id = random.sample(range(self.sample_num2), 1)[0]
            img_now = self.path2img(self.now_img[id])
            img_nxt = self.path2img(self.nxt_img[id])
            gt_now = self.now_gt[id]
            gt_nxt = self.nxt_gt[id]
            return (img_now, self.act_img[id], img_nxt), (gt_now, gt_nxt)
        except:
            print('no images')
            img_now = self.path2img(self.now_img[0])
            img_nxt = self.path2img(self.nxt_img[0])
            gt_now = self.now_gt[0]
            gt_nxt = self.nxt_gt[0]
            return (img_now,self.act_img[0],img_nxt), (gt_now,gt_nxt)

    def __getitem__(self, item):
        if self.opt.istrain:
            item1,gt_state = self.get_img_sample()
            item2 = self.get_state_sample()
            return item1,item2,gt_state
        else:
            item1, gt_state = self.get_img_sample(item)
            item2 = self.get_state_sample(item)
            return item1, item2, gt_state

    def __len__(self):
        return min(self.sample_num1,self.sample_num2)

    @classmethod
    def get_loader(cls,opt=None):
        dataset = cls(opt)
        return Data.DataLoader(
            dataset = dataset,
            batch_size = opt.batch_size,
            shuffle = opt.istrain,
            num_workers = 32
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/shared/xiaolonw/zqdata/cycle_nian/reallogs/second_version/exp/stop_only/Sphero-PYW/1',
                        help='Data directory to visualize')
    args = parser.parse_args()

    agent = BoltData(args.data_dir)