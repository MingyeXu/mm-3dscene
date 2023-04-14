import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset
import torch
from util.data_util import sa_create
from util.data_util import data_prepare


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def scene_crop(self,org_coord, org_feat, org_label, GT_feat):
        LEN_X = torch.max(org_coord[:,0]) - torch.min(org_coord[:,0])
        LEN_Y = torch.max(org_coord[:,1]) - torch.min(org_coord[:,1])
        
        CROP_X = LEN_X * (1/3 + np.random.rand()*2/3)/2
        CROP_Y = LEN_Y * (1/3 + np.random.rand()*2/3)/2

        t=torch.zeros(org_coord.shape[0])
        # crop_ids = []
        # for i in range(300):
            # crop_center = org_coord[np.random.randint(0,org_coord.shape[0]),:] 
        center_x = torch.min(org_coord[:,0]) + np.random.rand()* LEN_X
        center_y = torch.min(org_coord[:,1]) + np.random.rand()* LEN_Y



        crop_ids=(np.where((org_coord[:,0]<center_x+CROP_X) & (org_coord[:,0]>center_x-CROP_X)& (org_coord[:,1]<center_y+CROP_Y) & (org_coord[:,1]>center_y-CROP_Y) )[0])
    

        t[crop_ids]+=1
        save_ids = torch.where(t==0)[0]
        croped_coord = org_coord[save_ids]
        croped_feat = org_feat[save_ids]
        croped_feat_GT = GT_feat[save_ids]
        croped_label = org_label[save_ids]
        return croped_coord, croped_feat, croped_label, croped_feat_GT






    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label, _ = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        if self.split=='train':
            coord, feat, label, _ = self.scene_crop(coord, feat, label, _)
        return coord, feat, label
    def __len__(self):
        return len(self.data_idx) * self.loop
