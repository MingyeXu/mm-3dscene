import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from util.data_util import data_prepare
from util.data_util import sa_create
import SharedArray as SA
# import scipy.io as sio


class ScanNet_Grid(Dataset):
    def __init__(self, split='train', data_root='scannet', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        
        # data_file_org = '../data/scannet_orgdata'
        # data_file_partial = '../data/scannet_partialdata'
        # # if split =='train':
        # #     scene_names = np.loadtxt('../data/scannetv2_train.txt',dtype=np.str_)
        # # else:
        # #     scene_names = np.loadtxt('../data/scannetv2_val.txt',dtype=np.str_)
        # scene_names =np.concatenate([ np.loadtxt('../data/scannetv2_train.txt',dtype=np.str_) , np.loadtxt('../data/scannetv2_val.txt',dtype=np.str_) ])
        xyz_all=[]
        rgb_all=[]
        sem_labels_all =[]
        # ins_labels_all =[]

        partial_xyz_all=[]
        partial_rgb_all=[]
        partial_sem_labels_all =[]
        # partial_ins_labels_all =[]

        self.scene_names = []
        # for scene in scene_names:            
        #     if  not os.path.exists("/dev/shm/{}".format(scene+'_org')):
        #         print(os.path.exists("/dev/shm/{}".format(scene+'_org')))
        #         org_data = torch.load(os.path.join(data_file_org,scene+'.pth'))
        #         org_scene = np.concatenate((org_data[0],org_data[1],org_data[2].reshape(org_data[2].shape[0],1)), axis=1)
        #         if  not os.path.exists("/dev/shm/{}".format(scene+'_org')):
        #             try:    
        #                 sa_create("shm://{}".format(scene+'_org'), org_scene)
        #             except OSError:
        #                 pass
        #     self.scene_names.append(scene+'_org')
            
        s3dis_root = '../data/s3dis/trainval_fullarea'
        test_area = 5
        data_list = sorted(os.listdir(s3dis_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        for item in data_list:
            # if not os.path.exists("/dev/shm/{}".format(item)):
            data_path = os.path.join(s3dis_root, item + '.npy')
            data = np.load(data_path)  # xyzrgbl, N*7
            if  not os.path.exists("/dev/shm/{}".format(item+'_org')):
                try:    
                    sa_create("shm://{}".format(item+'_org'), data)
                except OSError:
                    pass                
            self.scene_names.append(item+'_org')





        # self.data_idx = np.arange(len(self.data_list))

        # self.xyz_all = xyz_all
        # self.rgb_all=rgb_all
        # self.sem_labels_all =sem_labels_all

        # self.partial_xyz_all = partial_xyz_all
        # self.partial_rgb_all=partial_rgb_all
        # self.partial_sem_labels_all =partial_sem_labels_all


        print("Totally {} samples in {} set.".format(len(self.scene_names), split))


    def scene_crop(self,org_coord, org_feat, org_label, GT_feat):
        LEN_X = torch.max(org_coord[:,0]) - torch.min(org_coord[:,0])
        LEN_Y = torch.max(org_coord[:,1]) - torch.min(org_coord[:,1])
        
        CROP_X = LEN_X/3
        CROP_Y = LEN_Y/3

        t=torch.zeros(org_coord.shape[0])
        crop_ids = []
        # for i in range(300):
            # crop_center = org_coord[np.random.randint(0,org_coord.shape[0]),:] 
        center_x = torch.min(org_coord[:,0]) + np.random.rand()* LEN_X
        center_y = torch.min(org_coord[:,1]) + np.random.rand()* LEN_Y
        # center_x = torch.min(org_coord[:,0]) + ((i*9)//30)* CROP_X
        # center_y = torch.min(org_coord[:,1]) + ((i*9)%30)* CROP_Y


        crop_ids=(np.where((org_coord[:,0]<center_x+CROP_X) & (org_coord[:,0]>center_x-CROP_X)& (org_coord[:,1]<center_y+CROP_Y) & (org_coord[:,1]>center_y-CROP_Y) )[0])
    
        # crop_ids = np.concatenate(crop_ids)
        # crop_ids = torch.cat(crop_ids,dim=0)
        t[crop_ids]+=1
        save_ids = torch.where(t==0)[0]
        croped_coord = org_coord[save_ids]
        croped_feat = org_feat[save_ids]
        croped_feat_GT = GT_feat[save_ids]
        croped_label = org_label[save_ids]
        return croped_coord, croped_feat, croped_label, croped_feat_GT




    def random_crop(self, org_coord, org_feat, org_label, GT_feat):
        # min_x = np.min(org_coord[:,0])
        # max_x = np.max(org_coord[:,0])
        # min_y = np.min(org_coord[:,1])
        # max_y = np.max(org_coord[:,1])
        # print(org_coord)
        LEN_X = torch.max(org_coord[:,0]) - torch.min(org_coord[:,0])
        LEN_Y = torch.max(org_coord[:,1]) - torch.min(org_coord[:,1])
        
        CROP_X = LEN_X/30
        CROP_Y = LEN_Y/30

        t=torch.zeros(org_coord.shape[0])
        crop_ids = []
        for i in range(300):
            # crop_center = org_coord[np.random.randint(0,org_coord.shape[0]),:] 
            center_x = torch.min(org_coord[:,0]) + np.random.randint(0,30)* CROP_X
            center_y = torch.min(org_coord[:,1]) + np.random.randint(0,30)* CROP_Y
            # center_x = torch.min(org_coord[:,0]) + ((i*9)//30)* CROP_X
            # center_y = torch.min(org_coord[:,1]) + ((i*9)%30)* CROP_Y


            crop_ids.append(np.where((org_coord[:,0]<center_x+CROP_X) & (org_coord[:,0]>center_x-CROP_X)& (org_coord[:,1]<center_y+CROP_Y) & (org_coord[:,1]>center_y-CROP_Y) )[0])
        crop_ids = np.concatenate(crop_ids)
        # crop_ids = torch.cat(crop_ids,dim=0)
        t[crop_ids]+=1
        save_ids = torch.where(t==0)[0]
        partial_coord = org_coord[save_ids]
        partial_feat = org_feat[save_ids]
        partial_feat_GT = GT_feat[save_ids]
        partial_label = org_label[save_ids]

        crop_ids = torch.where(t!=0)[0]
        crop_coord = org_coord[crop_ids]
        crop_feat = org_feat[crop_ids]
        crop_feat_GT = GT_feat[crop_ids]
        
        crop_label = org_label[crop_ids]

        # print('partial_coord',partial_coord.size())
        # print('org_coord',org_coord.size())
        # sio.savemat('sample_test.mat', {'partial_points':partial_coord.cpu().detach().numpy(), 'target_coor':org_coord.detach().cpu().numpy()})
        # exit(0)
        return partial_coord, partial_feat,partial_label,crop_coord, crop_feat, crop_label,partial_feat_GT,crop_feat_GT






    def __getitem__(self, idx):
        # data_idx = self.data_idx[idx % len(self.data_idx)]
        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        scene = self.scene_names[idx % len(self.scene_names)]


        # partial_data = SA.attach("shm://{}".format(scene)).copy()
        org_data = SA.attach("shm://{}".format(scene)).copy()
        # coord, feat, label = data
        # coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        
        # partial_coord, partial_feat, partial_label = partial_data[:,:3],partial_data[:,3:6],partial_data[:,6]
        # partial_coord, partial_feat, partial_label = data_prepare(partial_coord, partial_feat, partial_label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        org_coord, org_feat, org_label = org_data[:,:3],org_data[:,3:6],org_data[:,6]
        # print('before:',np.max(org_feat))
        # print('before:',np.min(org_feat))
        # print('org_feat0',org_feat.shape)
        org_coord, org_feat, org_label, GT_feat = data_prepare(org_coord, org_feat, org_label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        # print('org_feat0',org_feat.shape)
        # _, GT_feat, _ = data_prepare(org_coord, org_feat, org_label, self.split, self.voxel_size, self.voxel_max, None, self.shuffle_index)
        # print('GT_feat',GT_feat.shape)
        # print(torch.max(org_feat))
        # print(torch.min(org_feat))
        # exit(0)
        org_coord_1, org_feat_1, org_label_1, GT_feat_1 = self.scene_crop(org_coord, org_feat, org_label, GT_feat)
        partial_coord_1, partial_feat_1, partial_label_1, crop_coord_1, crop_feat_1, crop_label_1,partial_feat_GT_1,crop_feat_GT_1 = self.random_crop(org_coord_1, org_feat_1, org_label_1,GT_feat_1)
        # print(partial_coord.size())
        
        # org_coord_2, org_feat_2, org_label_2, GT_feat_2 = self.scene_crop(org_coord, org_feat, org_label, GT_feat)
        # partial_coord_2, partial_feat_2, partial_label_2, crop_coord_2, crop_feat_2, crop_label_2,partial_feat_GT_2,crop_feat_GT_2 = self.random_crop(org_coord_2, org_feat_2, org_label_2,GT_feat_2)


        return partial_coord_1, partial_feat_1, org_coord_1, org_feat_1
            #   partial_coord_2, partial_feat_2, org_coord_2
    def __len__(self):
        return len(self.scene_names)*self.loop

if __name__ == '__main__':
    train_data = ScanNet_Grid(split='train')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=None, num_workers=args.workers, pin_memory=True)
    for i, (coord, feat, target,target_coor, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        print(coord.size())



# if __name__ == '__main__':
#     data_root = '/mnt/sda1/hszhao/dataset/scannet'
#     point_data = ScanNet(split='train', data_root=data_root, num_point=8192, transform=None)
#     print('point data size:', point_data.__len__())
#     print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
#     print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
#     import torch, time, random
#     manual_seed = 123
#     def worker_init_fn(worker_id):
#         random.seed(manual_seed + worker_id)
#     random.seed(manual_seed)
#     np.random.seed(manual_seed)
#     torch.manual_seed(manual_seed)
#     torch.cuda.manual_seed_all(manual_seed)
#     train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
#     for idx in range(2):
#         end = time.time()
#         for i, (org_points, org_labels, partial_points, partial_labels ) in enumerate(train_loader):
#             print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
#             end = time.time()







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