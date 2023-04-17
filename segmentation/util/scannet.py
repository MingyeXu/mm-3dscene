import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SharedArray as SA
# import scipy.io as sio

from util.data_util import data_prepare, sa_create



remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

class ScanNet(Dataset):
    def __init__(self, split='train', data_root='scannet', voxel_size=0.02, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        
        # append scannet to shm
        # get data_name
        self.data_path = os.path.join(data_root, 'scannet_train_detection_data')
        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))



        if split =='all':         
            self.scan_names = all_scan_names
        elif split in ['train', 'val', 'test']:
            split_filenames = os.path.join(data_root, 'meta_data',
                'scannetv2_{}.txt'.format(split))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('illegal split name')

        # get data
        # for scan in self.scan_names:            
        #     if not os.path.exists("/dev/shm/{}".format(scan)):
        #         vert_path = os.path.join(self.data_path, scan + '_vert.npy')
        #         sem_path = os.path.join(self.data_path, scan + '_sem_label.npy')
        #         ins_path = os.path.join(self.data_path, scan + '_ins_label.npy')
        #         vert = np.load(vert_path)  # xyzrgb, N*6
        #         sem = np.load(sem_path)  # N,
        #         ins = np.load(ins_path)  # N,
        #         data = np.concatenate((vert, np.expand_dims(sem, axis=-1), np.expand_dims(ins, axis=-1)), axis=-1)  # npy, n*8
        #         sa_create("shm://{}".format(scan), data)



        self.data_idx = np.arange(len(self.scan_names))


        print("Totally {} samples in {} set.".format(len(self.data_idx), split))
    def scene_crop(self, org_coord, org_feat, org_label, GT_feat):
        LEN_X = torch.max(org_coord[:,0]) - torch.min(org_coord[:,0])
        LEN_Y = torch.max(org_coord[:,1]) - torch.min(org_coord[:,1])
        
        CROP_X = LEN_X * (1/3 + np.random.rand()*2/3)/2
        CROP_Y = LEN_Y * (1/3 + np.random.rand()*2/3)/2
        # CROP_X = LEN_X/5
        # CROP_Y = LEN_Y/5

        t=torch.zeros(org_coord.shape[0])
        crop_ids = []
        center_x = torch.min(org_coord[:,0]) + np.random.rand()* LEN_X
        center_y = torch.min(org_coord[:,1]) + np.random.rand()* LEN_Y
        crop_ids=(np.where((org_coord[:,0]<center_x+CROP_X) & (org_coord[:,0]>center_x-CROP_X)& (org_coord[:,1]<center_y+CROP_Y) & (org_coord[:,1]>center_y-CROP_Y) )[0])
        t[crop_ids]+=1
        save_ids = torch.where(t!=0)[0]
        croped_coord = org_coord[save_ids]
        croped_feat = org_feat[save_ids]
        croped_feat_GT = GT_feat[save_ids]
        croped_label = org_label[save_ids]
        return croped_coord, croped_feat, croped_label, croped_feat_GT

    
    def scene_crop_mutian(org_coord, org_feat, org_label, GT_feat, KEEP_RATIO, crop_dim):
        LEN_X = np.max(org_coord[:,0]) - np.min(org_coord[:,0])
        LEN_Y = np.max(org_coord[:,1]) - np.min(org_coord[:,1])
        LEN_Z = np.max(org_coord[:,2]) - np.min(org_coord[:,2])

        CROP_X = LEN_X * (KEEP_RATIO/2)
        CROP_Y = LEN_Y * (KEEP_RATIO/2)
        CROP_Z = LEN_Z * (KEEP_RATIO/2)

        # make sure the seleted area will not exceed the actual scene:
        if crop_dim == 0:
            center_x = random_range(np.min(org_coord[:,0])+CROP_X, np.max(org_coord[:,0])-CROP_X)
        elif crop_dim == 1:
            center_y = random_range(np.min(org_coord[:,1])+CROP_Y, np.max(org_coord[:,1])-CROP_Y)
        elif crop_dim == 2:
            center_z = random_range(np.min(org_coord[:,2])+CROP_Z, np.max(org_coord[:,2])-CROP_Z)
        else:
            raise Exception('cropped_dim not supported')

        if crop_dim == 0:
            save_ids = np.where((org_coord[:,0]<center_x+CROP_X) & (org_coord[:,0]>center_x-CROP_X))[0]
        elif crop_dim == 1:   
            save_ids = np.where((org_coord[:,1]<center_y+CROP_Y) & (org_coord[:,1]>center_y-CROP_Y))[0]
        elif crop_dim == 2:
            save_ids = np.where((org_coord[:,2]<center_z+CROP_Z) & (org_coord[:,2]>center_z-CROP_Z))[0]
        croped_coord = org_coord[save_ids]
        croped_feat = org_feat[save_ids]
        croped_feat_GT = GT_feat[save_ids]
        croped_label = org_label[save_ids]
        return croped_coord, croped_feat, croped_label, croped_feat_GT



    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        org_data = SA.attach("shm://{}".format(self.scan_names[data_idx])).copy()
        org_coord, org_feat, org_label = org_data[:,:3],org_data[:,3:6],org_data[:,6]
        org_label = remapper[org_label.astype(int)]
        # print('before:',np.max(org_feat))
        # print('before:',np.min(org_feat))
        # print('org_feat0',org_feat.shape)
        org_coord, org_feat, org_label, GT_feat = data_prepare(org_coord, org_feat, org_label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        Croped_scene = self.scene_crop(org_coord, org_feat, org_label, GT_feat)
        if Croped_scene[0].shape[0]<512:
            org_coord_1, org_feat_1, org_label_1, GT_feat_1 = org_coord, org_feat, org_label, GT_feat
        else:
            org_coord_1, org_feat_1, org_label_1, GT_feat_1 = Croped_scene

        # partial_coord_1, partial_feat_1, partial_label_1, crop_coord_1, crop_feat_1, crop_label_1,partial_feat_GT_1,crop_feat_GT_1 = self.random_crop(org_coord_1, org_feat_1, org_label_1,GT_feat_1)

        return  org_coord_1, org_feat_1, org_label_1


    # def __getitem__(self, idx):
    #     data_idx = self.data_idx[idx % len(self.data_idx)]
    #     org_data = SA.attach("shm://{}".format(self.scan_names[data_idx])).copy()
    #     org_coord, org_feat, org_label = org_data[:,:3],org_data[:,3:6],org_data[:,6]
    #     org_coord, org_feat, org_label, GT_feat = data_prepare(org_coord, org_feat, org_label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

    #     Croped_scene = self.scene_crop(org_coord, org_feat, org_label, GT_feat)
    #     if Croped_scene[0].shape[0]<512:
    #         org_coord_1, org_feat_1, org_label_1, GT_feat_1 = org_coord, org_feat, org_label, GT_feat
    #     else:
    #         org_coord_1, org_feat_1, org_label_1, GT_feat_1 = Croped_scene

    #     # partial_coord_1, partial_feat_1, partial_label_1, crop_coord_1, crop_feat_1, crop_label_1,partial_feat_GT_1,crop_feat_GT_1 = self.random_crop(org_coord_1, org_feat_1, org_label_1,GT_feat_1)

    #     return  org_coord_1, org_feat_1, org_label_1

    def __len__(self):
        return len(self.data_idx)*self.loop


if __name__ == '__main__':
    data_root = '/mnt/sda1/hszhao/dataset/scannet'
    point_data = ScanNet(split='train', data_root=data_root, num_point=8192, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(2):
        end = time.time()
        for i, (org_points, org_labels, partial_points, partial_labels ) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()







        # label_all = []  # for change 0-20 to 0-19 + 255
        # partial_label_all = []
        # room_coord_min, room_coord_max = [], []
        # num_point_all = []
        # label_weight = np.zeros(classes+1)
        # for index in range(len(xyz_all)):
        #     xyz, label = xyz_all[index], sem_labels_all[index]  # xyz, N*3; l, N
        #     partial_xyz ,partial_label = partial_xyz_all[index], partial_sem_labels_all[index]
        #     coord_min, coord_max = np.amin(xyz, axis=0)[:3], np.amax(xyz, axis=0)[:3]
        #     room_coord_min.append(coord_min), room_coord_max.append(coord_max)
        #     num_point_all.append(label.size)
        #     tmp, _ = np.histogram(label, range(classes + 2))
        #     label_weight += tmp
        #     label_new = label - 1
        #     label_new[label == 0] = 255
        #     label_all.append(label_new.astype(np.uint8))
        #     partial_label_new = partial_label - 1
        #     partial_label_new[partial_label == 0] = 255
        #     partial_label_all.append(partial_label_new.astype(np.uint8))

        # label_weight = label_weight[1:].astype(np.float32)
        # label_weight = label_weight / label_weight.sum()
        # label_weight = 1 / np.log(1.2 + label_weight)
        # sample_prob = num_point_all / np.sum(num_point_all)
        # num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        # room_idxs = []
        # for index in range(len(xyz_all)):
        #     room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        # room_idxs = np.array(room_idxs)
        # xyz_all = xyz_all
        # partial_xyz_all = partial_xyz_all
        # label_weight = label_weight





        # room_idx = room_idxs[idx]
        # points = xyz_all[room_idx]  # N * 3
        # partial_points = partial_xyz_all[room_idx]
        # labels = label_all[room_idx]  # N
        # partial_labels = partial_label_all[room_idx]
        # N_points = points.shape[0]

        # for i in range(10):
        #     center = points[np.random.choice(N_points)][:3]
        #     block_min = center - [block_size / 2.0, block_size / 2.0, 0]
        #     block_max = center + [block_size / 2.0, block_size / 2.0, 0]
        #     block_min[2], block_max[2] = room_coord_min[room_idx][2], room_coord_max[room_idx][2]
        #     point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
        #     partial_idxs = np.where((partial_points[:, 0] >= block_min[0]) & (partial_points[:, 0] <= block_max[0]) & (partial_points[:, 1] >= block_min[1]) & (partial_points[:, 1] <= block_max[1]))[0]

        #     if point_idxs.size == 0:
        #         continue
        #     vidx = np.ceil((points[point_idxs, :] - block_min) / (block_max - block_min) * [31.0, 31.0, 62.0])
        #     vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
        #     if ((labels[point_idxs] != 255).sum() / point_idxs.size >= 0.7) and (vidx.size/31.0/31.0/62.0 >= 0.02):
        #         break

        # if point_idxs.size >= num_point:
        #     selected_point_idxs = np.random.choice(point_idxs, num_point, replace=False)
        #     selected_partial_point_idxs = np.random.choice(partial_idxs, num_point, replace=False)
        # else:
        #     selected_point_idxs = np.random.choice(point_idxs, num_point, replace=True)
        #     selected_partial_point_idxs = np.random.choice(partial_idxs, num_point, replace=True)
        # # normalize
        # selected_points = points[selected_point_idxs, :]  # num_point * 3
        # selected_partial_points = partial_points[selected_partial_point_idxs, :]  # num_point * 3
        # current_points = np.zeros((num_point, 6))  # num_point * 6
        # current_points[:, 3] = selected_points[:, 0] / room_coord_max[room_idx][0]
        # current_points[:, 4] = selected_points[:, 1] / room_coord_max[room_idx][1]
        # current_points[:, 5] = selected_points[:, 2] / room_coord_max[room_idx][2]
        # selected_points[:, 0] = selected_points[:, 0] - center[0]
        # selected_points[:, 1] = selected_points[:, 1] - center[1]
        # current_points[:, 0:3] = selected_points

        # current_partial_points = np.zeros((num_point, 6))  # num_point * 6
        # current_partial_points[:, 3] = selected_partial_points[:, 0] / room_coord_max[room_idx][0]
        # current_partial_points[:, 4] = selected_partial_points[:, 1] / room_coord_max[room_idx][1]
        # current_partial_points[:, 5] = selected_partial_points[:, 2] / room_coord_max[room_idx][2]
        # selected_partial_points[:, 0] = selected_partial_points[:, 0] - center[0]
        # selected_partial_points[:, 1] = selected_partial_points[:, 1] - center[1]
        # current_partial_points[:, 0:3] = selected_partial_points

        # current_labels = labels[selected_point_idxs]
        # current_partial_labels = partial_labels[selected_partial_point_idxs]