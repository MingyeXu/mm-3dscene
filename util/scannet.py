import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ScanNet(Dataset):
    def __init__(self, split='train', data_root='scannet', num_point=8192, classes=20, block_size=1.5, sample_rate=1.0, transform=None):
        self.split = split
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        data_file_org = '/opt/data/private/scene/data/scannet_orgdata'
        data_file_partial = '/opt/data/private/scene/data/scannet_partialdata'
        if split =='train':
            scene_names = np.loadtxt('/opt/data/private/scene/data/scannetv2_train.txt',dtype=np.str_)
        else:
            scene_names = np.loadtxt('/opt/data/private/scene/data/scannetv2_val.txt',dtype=np.str_)
        
        xyz_all=[]
        rgb_all=[]
        sem_labels_all =[]
        # ins_labels_all =[]

        partial_xyz_all=[]
        partial_rgb_all=[]
        partial_sem_labels_all =[]
        # partial_ins_labels_all =[]

        for scene in scene_names:
            data = torch.load(os.path.join(data_file_org,scene+'.pth'))
            partial_data = torch.load(os.path.join(data_file_partial,scene+'.pth.pth'))
            # keeped_coords, keeped_colors, keeped_sem_labels, keeped_instance_labels
            xyz_all.append(data[0])
            rgb_all.append(data[1])
            sem_labels_all.append(data[2])
            partial_xyz_all.append(partial_data[0])
            partial_rgb_all.append(partial_data[1])
            partial_sem_labels_all.append(partial_data[2])

        # data_file = os.path.join(data_root, 'scannet_{}.pickle'.format(split))
        # file_pickle = open(data_file, 'rb')
        # xyz_all = pickle.load(file_pickle, encoding='latin1')
        # label_all = pickle.load(file_pickle, encoding='latin1')
        # file_pickle.close()

        # label_all = []  # for change 0-20 to 0-19 + 255
        # room_coord_min, room_coord_max = [], []
        # num_point_all = []
        # label_weight = np.zeros(classes+1)
        # for index in range(len(xyz_all)):
        #     xyz, label = xyz_all[index], sem_labels_all[index]  # xyz, N*3; l, N
        #     coord_min, coord_max = np.amin(xyz, axis=0)[:3], np.amax(xyz, axis=0)[:3]
        #     room_coord_min.append(coord_min), room_coord_max.append(coord_max)
        #     num_point_all.append(label.size)
        #     tmp, _ = np.histogram(label, range(classes + 2))
        #     label_weight += tmp
        #     label_new = label - 1
        #     label_new[label == 0] = 255
        #     label_all.append(label_new.astype(np.uint8))




        self.label_all = []  # for change 0-20 to 0-19 + 255
        self.partial_label_all = []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        label_weight = np.zeros(classes+1)
        for index in range(len(xyz_all)):
            xyz, label = xyz_all[index], sem_labels_all[index]  # xyz, N*3; l, N
            partial_xyz ,partial_label = partial_xyz_all[index], partial_sem_labels_all[index]
            coord_min, coord_max = np.amin(xyz, axis=0)[:3], np.amax(xyz, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(label.size)
            tmp, _ = np.histogram(label, range(classes + 2))
            label_weight += tmp
            label_new = label - 1
            label_new[label == 0] = 255
            self.label_all.append(label_new.astype(np.uint8))
            partial_label_new = partial_label - 1
            partial_label_new[partial_label == 0] = 255
            self.partial_label_all.append(partial_label_new.astype(np.uint8))

        label_weight = label_weight[1:].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = 1 / np.log(1.2 + label_weight)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(xyz_all)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        self.xyz_all = xyz_all
        self.partial_xyz_all = partial_xyz_all
        self.label_weight = label_weight
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.xyz_all[room_idx]  # N * 3
        partial_points = self.partial_xyz_all[room_idx]
        labels = self.label_all[room_idx]  # N
        partial_labels = self.partial_label_all[room_idx]
        N_points = points.shape[0]

        for i in range(10):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_min[2], block_max[2] = self.room_coord_min[room_idx][2], self.room_coord_max[room_idx][2]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            partial_idxs = np.where((partial_points[:, 0] >= block_min[0]) & (partial_points[:, 0] <= block_max[0]) & (partial_points[:, 1] >= block_min[1]) & (partial_points[:, 1] <= block_max[1]))[0]

            if point_idxs.size == 0:
                continue
            if partial_idxs.size == 0:
                continue
            
            vidx = np.ceil((points[point_idxs, :] - block_min) / (block_max - block_min) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            if ((labels[point_idxs] != 255).sum() / point_idxs.size >= 0.7) and (vidx.size/31.0/31.0/62.0 >= 0.02):
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            # selected_partial_point_idxs = np.random.choice(partial_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
            # selected_partial_point_idxs = np.random.choice(partial_idxs, self.num_point, replace=True)

        if partial_idxs.size >= self.num_point:
            # selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            selected_partial_point_idxs = np.random.choice(partial_idxs, self.num_point, replace=False)
        else:
            
            if partial_idxs.shape[0]==0:
                # print(partial_idxs.shape)
                selected_partial_point_idxs = selected_point_idxs
                partial_points = points
                partial_labels = labels
            else:
                # selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
                selected_partial_point_idxs = np.random.choice(partial_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 3
        selected_partial_points = partial_points[selected_partial_point_idxs, :]  # num_point * 3
        current_points = np.zeros((self.num_point, 6))  # num_point * 6
        current_points[:, 3] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 4] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 5] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:3] = selected_points

        current_partial_points = np.zeros((self.num_point, 6))  # num_point * 6
        current_partial_points[:, 3] = selected_partial_points[:, 0] / self.room_coord_max[room_idx][0]
        current_partial_points[:, 4] = selected_partial_points[:, 1] / self.room_coord_max[room_idx][1]
        current_partial_points[:, 5] = selected_partial_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_partial_points[:, 0] = selected_partial_points[:, 0] - center[0]
        selected_partial_points[:, 1] = selected_partial_points[:, 1] - center[1]
        current_partial_points[:, 0:3] = selected_partial_points

        current_labels = labels[selected_point_idxs]
        current_partial_labels = partial_labels[selected_partial_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
            current_partial_points, current_partial_labels = self.transform(current_partial_points, current_partial_labels)
        return current_points, current_labels ,current_partial_points,current_partial_labels

    def __len__(self):
        return len(self.room_idxs)


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