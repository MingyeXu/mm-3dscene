import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x
    

def collate_fn(batch):
    partial_coord_1, partial_feat_1, org_coord_1, org_feat_1 = list(zip(*batch))
    offset_par_1, count_par_1 = [], 0
    offset_org_1, count_org_1 = [], 0

    for item in partial_coord_1:
        count_par_1 += item.shape[0]
        offset_par_1.append(count_par_1)
    for item in org_coord_1:
        count_org_1 += item.shape[0]
        offset_org_1.append(count_org_1)


    return torch.cat(partial_coord_1), torch.cat(partial_feat_1), torch.cat(org_coord_1), torch.cat(org_feat_1), \
        torch.IntTensor(offset_par_1),torch.IntTensor(offset_org_1)
# def collate_fn(batch):
#     org_coord, org_feat, org_label, GT_feat, mask_label = list(zip(*batch))
#     offset, count = [], 0
#     # offset2, count2 = [], 0
#     # offset3, count3 = [], 0
#     # new_coord = []
#     # new_target = []
#     for item in org_coord:
#         count += item.shape[0]
#         offset.append(count)


#     return torch.cat(org_coord), torch.cat(org_feat), torch.cat(org_label),\
#         torch.cat(GT_feat),torch.cat(mask_label), torch.IntTensor(offset)


# def collate_fn(batch):
#     coord, feat, label, target_points,  target_feat,GT_feat, crop_coord, crop_feat = list(zip(*batch))
#     offset, count = [], 0
#     offset2, count2 = [], 0
#     offset3, count3 = [], 0
#     new_coord = []
#     new_target = []
#     for item in coord:
#         count += item.shape[0]
#         offset.append(count)
#         # coor_item = (item / item.max(dim =0)[0]) - 0.5
#         # new_coord.append(coor_item)
#     for item in target_points:
#         count2 += item.shape[0]
#         offset2.append(count2)
#     for item in crop_coord:
#         count3 += item.shape[0]
#         offset3.append(count3)
#         # coor_item = (item / item.max(dim =0)[0]) - 0.5
#         # new_target.append(coor_item)
#     # a = torch.cat(coord)
#     # a = torch.cat(feat)
#     # a = torch.cat(label)
#     # a = torch.cat(targetpoints)
    
#     # a = torch.IntTensor(offset)
#     return torch.cat(coord), torch.cat(feat), torch.cat(label),\
#         torch.cat(target_points), torch.cat(target_feat),torch.cat(GT_feat), \
#         torch.cat(crop_coord), torch.cat(crop_feat),\
#         torch.IntTensor(offset),torch.IntTensor(offset2),torch.IntTensor(offset3)



def collate_fn_s3dis(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    # a = torch.cat(coord)
    # a = torch.cat(feat)
    # a = torch.cat(label)
    # a = torch.cat(targetpoints)
    
    # a = torch.IntTensor(offset)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)




def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if np.min(feat)<0:
        feat = (feat+1)*255./2.
    if np.min(coord)<0:
        coord = coord-np.min(coord,axis=0)

    # print('@@',np.max(coord))
    # print('@@',np.min(coord))

    org_feat = feat.copy()
    if transform:
        # print('feat')
        
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, axis=0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label, org_feat = coord[uniq_idx], feat[uniq_idx], label[uniq_idx],org_feat[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label, org_feat = coord[crop_idx], feat[crop_idx], label[crop_idx], org_feat[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label, org_feat = coord[shuf_idx], feat[shuf_idx], label[shuf_idx], org_feat[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)

    # if np.max(feat)>2:
    feat = torch.FloatTensor(feat) / 255.
    org_feat = torch.FloatTensor(org_feat) / 255.
    # else:
        # feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label, org_feat
