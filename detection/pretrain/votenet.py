# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss
import math
import pointnet2_utils
import torch.nn.functional as F
import scipy.io as sio

from lib.pointops.functions import pointops

from model_utils import *

from functools import wraps
import copy


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val



# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)







def cal_loss( pred_xyz, gt_xyz):

    loss,_ = calc_cd(pred_xyz,gt_xyz)

    return loss.mean()


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx





def get_graph_feature(x, rgb, k=30, idx=None):

    x = x.permute(0,2,1).contiguous()
    # rgb = rgb.permute(0,2,1).contiguous()    
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    rgb = rgb.transpose(2, 1).contiguous()
    rgb_knn = rgb.view(batch_size*num_points, -1)[idx, :]
    rgb_knn = rgb_knn.view(batch_size, num_points, k, num_dims) 
    rgb = rgb.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature-x, rgb_knn - rgb



def _break_up_pc(pc):
    xyz = pc[..., 0:3].contiguous()
    features = (
        pc[..., 3:].transpose(1, 2).contiguous()
        if pc.size(-1) > 3 else None
    )

    return xyz, features





def  mask_lowlevel_Diff(pxro, feat, mask_rate):

    p,x,r,o = pxro
    C = x.size(-1)
    # x =  torch.cat((p0, x0), 1)
    nsample = 30

    knn_x = pointops.queryandgroup(nsample, p, p, x, None, o, o, use_xyz=False) 
    center_x = x.reshape(-1,1,C).repeat(1,nsample,1)
    d_x = ((center_x - knn_x)**2).sum(-1).sum(-1) # N
    min_d = d_x.min(dim=-1)[0].repeat([d_x.size(0)])
    max_d = d_x.max(dim=-1)[0].repeat([d_x.size(0)])
    d_x = (d_x-min_d)/(max_d-min_d)

    knn_r = pointops.queryandgroup(nsample, p, p, r, None, o, o, use_xyz=False) 
    center_r = r.reshape(-1,1,3).repeat(1,nsample,1)
    d_r = ((center_r - knn_r)**2).sum(-1).sum(-1) # N
    min_d = d_r.min(dim=-1)[0].repeat([d_r.size(0)])
    max_d = d_r.max(dim=-1)[0].repeat([d_r.size(0)])
    d_r = (d_r-min_d)/(max_d-min_d)


    knn_p = pointops.queryandgroup(nsample, p, p, p, None, o, o, use_xyz=False) 
    center_p = p.reshape(-1,1,3).repeat(1,nsample,1)
    d_p = ((center_p - knn_p)**2).sum(-1).sum(-1)
    min_d = d_p.min(dim=-1)[0].repeat([d_p.size(0)])
    max_d = d_p.max(dim=-1)[0].repeat([d_p.size(0)])
    d_p = (d_p-min_d)/(max_d-min_d)


    d = d_x + d_r + d_p   # consider all!
    # print(d.size())
    mask_p=[]
    mask_x=[]
    mask_o=[]
    count=0

    gt_p=[]
    gt_x=[]
    gt_o=[]
    gt_count=0



    for i in range(o.size(0)):
        p_i = p[o[i-1]:o[i],:] if i!=0  else p[:o[i],:]
        x_i = feat[o[i-1]:o[i],:] if i!=0  else feat[:o[i],:]
        d_i = d[o[i-1]:o[i]] if i!=0  else d[:o[i]]

        idx = d_i.topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
        mask_p.append(p_i[idx])
        mask_x.append(x_i[idx])
        count += idx.size(0)
        mask_o.append(count)

        gt_rate= mask_rate +0.1
        gt_idx = d_i.topk(k=int(gt_rate*p_i.size(0)), dim=0)[1]
        gt_p.append(p_i[gt_idx])
        gt_x.append(x_i[gt_idx])
        gt_count += gt_idx.size(0)
        gt_o.append(gt_count)



    mask_p = torch.cat(mask_p)
    mask_x = torch.cat(mask_x)
    mask_o = torch.IntTensor(mask_o).to(p.device)


    gt_p = torch.cat(gt_p)
    gt_x = torch.cat(gt_x)
    gt_o = torch.IntTensor(gt_o).to(p.device)


    return [mask_p,mask_x,mask_o], [gt_p, gt_x, gt_o]


def mask_fuc(pc, mask_rate, scene_embedding):


    xyz,feat = _break_up_pc(pc)
    rgb = feat[:,:3,:].permute(0,2,1).contiguous()
    pc = pc.permute(0,2,1).contiguous()
    f = feat.permute(0,2,1).contiguous()
    # print('feat',feat.size())


    B,N,_ = xyz.size()
    xyz = xyz.reshape(B*N,-1)
    rgb = rgb.reshape(B*N,-1)
    pc = pc.reshape(B*N,-1)
    f = f.reshape(B*N,-1)
    off = []
    for i in range(B):
        off.append(N*(i+1))
    off = torch.IntTensor(off).to(xyz.device)

    feat = scene_embedding([xyz,pc,off])  # use whole pc to compute feat diff

    mask_pxo, gt_pxo = mask_lowlevel_Diff([xyz,feat,rgb,off], f, mask_rate)

    mask_xyz = mask_pxo[0].reshape(B,mask_pxo[2][0],-1)
    mask_feat =  mask_pxo[1].reshape(B,mask_pxo[2][0],-1)

    gt_xyz = gt_pxo[0].reshape(B,gt_pxo[2][0],-1)
    gt_feat = gt_pxo[1].reshape(B,gt_pxo[2][0],-1)

    return mask_xyz, mask_feat, gt_xyz, gt_feat

# def  mask_lowlevel_Diff(pc, mask_rate):

#     # print(pc.size())
#     # p,x,o = pxo
#     xyz,feat = _break_up_pc(pc)
#     rgb = feat[:,:3,:]
#     # x =  torch.cat((p0, x0), 1)
#     # print(rgb[0,:,0:6])
#     # exit(0)

#     nsample = 30
    
#     d_x,d_p = get_graph_feature(xyz,rgb,nsample)    
#     # print(d_x.size())
#     # d_x = ((center_x - knn_x)**2)
#     d_x = (d_x**2).sum(-1).sum(-1) # B,N
#     min_d = torch.min(d_x,dim=-1, keepdim=True)[0].repeat([1,d_x.size(1)])
#     max_d = torch.max(d_x,dim=-1, keepdim=True)[0].repeat([1,d_x.size(1)])
#     # print(d_x.size())
#     # print(min_d.size())
#     d_x = (d_x-min_d)/(max_d-min_d)


#     # d_p = ((center_p - knn_p)**2)
#     d_p = (d_p**2).sum(-1).sum(-1)
#     min_d = torch.min(d_p,dim=-1, keepdim=True)[0].repeat([1,d_p.size(1)])
#     max_d = torch.max(d_p,dim=-1, keepdim=True)[0].repeat([1,d_p.size(1)])
#     d_p = (d_p-min_d)/(max_d-min_d)


#     d = d_x + d_p

#     # print(d.size())
#     # mask_rate = 0.2 + 0.6*torch.rand(1)
#     # print(int(mask_rate*d.size(1)))
#     idx = d.topk(k=int(mask_rate*d.size(1)), dim=-1)[1]

#     is_small_scene = False
#     if int(mask_rate*d.size(1))<512:
#         mask_rate = 0.8
#         idx = d.topk(k=int(mask_rate*d.size(1)), dim=-1)[1]
#         is_small_scene = True

#     # print(idx.size())

#     mask_xyz = pointnet2_utils.gather_operation(
#         xyz.transpose(1, 2).contiguous(), idx.int()
#     ).transpose(1, 2).contiguous() # B.M,3
#     # print('CCC',mask_xyz.size())
#     # exit(0)
#     mask_feat = pointnet2_utils.gather_operation(
#         feat, idx.int()
#     ).transpose(1, 2).contiguous()
#     # mask_xyz = xyz[idx] 
#     # print('CCC',mask_feat.size())
#     # exit(0)



#     if is_small_scene:
#         gt_xyz = xyz
#     else:
#         gt_rate= mask_rate +0.2
#         gt_idx = d.topk(k=int(gt_rate*d.size(1)), dim=-1)[1]
#         gt_xyz = pointnet2_utils.gather_operation(
#             xyz.transpose(1, 2).contiguous(), gt_idx.int()
#         ).transpose(1, 2).contiguous() # B.M,3

#     return mask_xyz, mask_feat, gt_xyz



class Fold(nn.Module):
    def __init__(self, in_channel, step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        sqrted = int(math.sqrt(step)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (step % i) == 0:
                num_x = i
                num_y = step // i
                break

        grid_x = torch.linspace(-0.2, 0.2, steps=num_x)
        grid_y = torch.linspace(-0.2, 0.2, steps=num_y)
        x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
        self.grid = torch.stack([x, y], dim=-1).view(-1, 2)  # (2, 2)



        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        # num_sample = self.step * self.step
        bs = x.size(0)
        num_points = x.size(2)
        # features = x
        # seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)
        features = x.repeat(1, 1, self.step)
        seed = self.grid.unsqueeze(0).repeat(bs, num_points, 1).transpose(1, 2).contiguous().cuda()
        # print('features',features.size())
        # print('seed',seed.size())
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        # print(fd2)
        return fd2 #B,3,N



class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling


        self.scene_embedding = PointTransformerLayer(in_planes=7, out_planes=16)


        # Backbone point feature learning
        self.target_ema_updater = EMA(0.99)
        self.target_encoder = None
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)

        self.expansion = Fold(256, 2 )



    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.backbone_net)
        set_requires_grad(target_encoder, False)
        return target_encoder


    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        # assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.backbone_net)

    def forward(self, inputs,mask_rate):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        mask_xyz, mask_feat, gt_xyz, gt_feat = mask_fuc(inputs['point_clouds'],mask_rate,self.scene_embedding)

        end_points = self.backbone_net(torch.cat([mask_xyz,mask_feat],dim=-1), end_points)
        

        # print('fp2_features',end_points['fp2_features'].size()) # B,C,1024
        # print('fp2_xyz',end_points['fp2_xyz'].size()) # B,1024,3
        # print('mask_xyz',mask_xyz.size())
        # exit(0)
        completion_pred = self.expansion(end_points['fp2_features']).permute(0,2,1).contiguous() 
        completion_pred += end_points['fp2_xyz'].repeat(1, 2, 1)


        with torch.no_grad():
            end_points_gt = {}
            target_encoder = self._get_target_encoder()
            end_points_gt = target_encoder(torch.cat([gt_xyz,gt_feat],dim=-1), end_points_gt)
            # print('fp2_features',end_points_gt['fp2_features'].size()) # B,C,1024
            # print('fp2_xyz',end_points_gt['fp2_xyz'].size()) # B,1024,3


            unknown = end_points['fp2_xyz']
            known = end_points_gt['fp2_xyz']
            known_feats = end_points_gt['fp2_features']
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            contrast_part_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )

        return completion_pred, gt_xyz, gt_feat, contrast_part_feats.permute(0,2,1).contiguous(), end_points['fp2_features'].permute(0,2,1).contiguous()






if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model
    model = VoteNet(10,12,10,np.random.random((10,3))).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key])

    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')
