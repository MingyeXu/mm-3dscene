import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from util.data_util import data_prepare
from util.data_util import sa_create
import SharedArray as SA
# import scipy.io as sio


class DatasetPretrain(Dataset):
    def __init__(self, split='train', pretrain_dataset='s3dis', data_root='', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        
        xyz_all=[]
        rgb_all=[]
        sem_labels_all =[]

        partial_xyz_all=[]
        partial_rgb_all=[]
        partial_sem_labels_all =[]

        self.scene_names = []
        self.pretrain_dataset = pretrain_dataset
        if self.pretrain_dataset == 's3dis':
            test_area = 5
            data_list = sorted(os.listdir(data_root))
            data_list = [item[:-4] for item in data_list if 'Area_' in item]
            data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
            for item in data_list:
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                if  not os.path.exists("/dev/shm/{}".format(item+'_org')):
                    try:    
                        sa_create("shm://{}".format(item+'_org'), data)
                    except OSError:
                        pass                
                self.scene_names.append(item+'_org')
            print("Totally {} samples in {} set.".format(len(self.scene_names), split))

        if self.pretrain_dataset == 'scannet':
            data_path = os.path.join(data_root, 'scannet_train_detection_data_wNormals')
            all_scan_names = list(set([os.path.basename(x)[0:12] \
                for x in os.listdir(data_path) if x.startswith('scene')]))
            split_filenames = os.path.join(data_path, 'meta_data',
                'scannetv2_{}.txt'.format('train'))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)

            # get data
            for scan in self.scan_names:            
                if not os.path.exists("/dev/shm/{}".format(scan)):
                    vert_path = os.path.join(data_path, scan + '_vert.npy')
                    sem_path = os.path.join(data_path, scan + '_sem_label.npy')
                    ins_path = os.path.join(data_path, scan + '_ins_label.npy')
                    vert = np.load(vert_path)  # xyzrgb, N*6
                    sem = np.load(sem_path)  # N,
                    ins = np.load(ins_path)  # N,
                    data = np.concatenate((vert, np.expand_dims(sem, axis=-1), np.expand_dims(ins, axis=-1)), axis=-1)  # npy, n*8
                    sa_create("shm://{}".format(scan), data)
                self.scene_names.append(item)

            # self.data_idx = np.arange(len(self.scan_names))
            print("Totally {} samples in {} set.".format(len(self.scene_names), split))





    def scene_crop(self,org_coord, org_feat, org_label, GT_feat):
        LEN_X = torch.max(org_coord[:,0]) - torch.min(org_coord[:,0])
        LEN_Y = torch.max(org_coord[:,1]) - torch.min(org_coord[:,1])
        
        CROP_X = LEN_X * (1/3 + np.random.rand()*2/3)/2
        CROP_Y = LEN_Y * (1/3 + np.random.rand()*2/3)/2

        t=torch.zeros(org_coord.shape[0])
        crop_ids = []
        center_x = torch.min(org_coord[:,0]) + np.random.rand()* LEN_X
        center_y = torch.min(org_coord[:,1]) + np.random.rand()* LEN_Y

        crop_ids=(np.where((org_coord[:,0]<center_x+CROP_X) & (org_coord[:,0]>center_x-CROP_X)& (org_coord[:,1]<center_y+CROP_Y) & (org_coord[:,1]>center_y-CROP_Y) )[0])
        t[crop_ids]+=1
        save_ids = torch.where(t==0)[0]
        croped_coord = org_coord[save_ids]
        croped_feat = org_feat[save_ids]
        croped_feat_GT = GT_feat[save_ids]
        croped_label = org_label[save_ids]

        if croped_coord.shape[0]<1024:
            return org_coord, org_feat, org_label, GT_feat
        else:
            return croped_coord, croped_feat, croped_label, croped_feat_GT


    def __getitem__(self, idx):
        scene = self.scene_names[idx % len(self.scene_names)]
        org_data = SA.attach("shm://{}".format(scene)).copy()
        org_coord, org_feat, org_label = org_data[:,:3],org_data[:,3:6],org_data[:,6]
        org_coord, org_feat, org_label, GT_feat = data_prepare(org_coord, org_feat, org_label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        org_coord, org_feat, org_label, GT_feat = self.scene_crop(org_coord, org_feat, org_label, GT_feat)
        return  org_coord, org_feat

    def __len__(self):
        return len(self.scene_names)*self.loop

if __name__ == '__main__':
    train_data = DatasetPretrain(split='train')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=None, num_workers=args.workers, pin_memory=True)
    for i, (coord, feat, target,target_coor, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        print(coord.size())

