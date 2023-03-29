from shutil import copy2
from ssl import PROTOCOL_TLSv1_1
from tkinter import S
import torch
import torch.nn as nn

from lib.pointops.functions import pointops
from model.pointtransformer.model_utils import *
from model.pointtransformer.SPD_refine import *
import numpy as np
import scipy.io as sio
# from lib.chamfer_dist import ChamferDistanceL1
from model.pointtransformer.position_embedding import *

from functools import wraps
import copy


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






def reduce_pointcloud(pt1,pt2,o1,o2):

    # print('pt1',pt1.size())
    # print('pt2',pt2.size())
    # exit(0)
    # o = torch.Tensor([0,pt1.size(0)//2]).cuda().int()
    knn = pointops.queryandgroup(1, pt2, pt1, pt2, None, o2, o1, use_xyz=True)
    dist = torch.sum(torch.abs((knn[:,:,0:3]**2)),dim=-1)
    # print('dist',dist.size())
    # exit(0)
    # print(dist>dist.mean())
    # exit(0)
    idx = torch.where(dist>dist.mean())[0]
    # idx = torch.where(dist>0.04)[0]

    # print('idx',idx.size())
    if idx.size(0)>0:
        reduce_pt = pt1[idx]
    else:
        reduce_pt = pt1
    # sio.savemat('reduce_test.mat', {'points':reduce_pt.cpu().detach().numpy(),'pt1':pt1.cpu().detach().numpy(),'pt2':pt2.cpu().detach().numpy()})
    # print('asd')
    # exit(0)
    return reduce_pt

def reduce_pointcloud_single(pt1,pt2):
    
    o1 = torch.Tensor([pt1.size(0)]).cuda().int()
    o2 = torch.Tensor([pt2.size(0)]).cuda().int()
    knn = pointops.queryandgroup(1, pt2, pt1, pt2, None, o2, o1, use_xyz=True)
    dist = torch.sum(torch.abs((knn[:,:,0:3]**2)),dim=-1)

    # idx = torch.where(dist>dist.mean())[0]
    idx = torch.where(dist>0.02)[0]
    # print('idx',idx.size())

    if idx.size(0)>0:
        reduce_pt = pt1[idx]
    else:
        reduce_pt = pt1
    return reduce_pt

def intersection_pointcloud(pt1, pt2):

    o1 = torch.Tensor([pt1.size(0)]).cuda().int()
    o2 = torch.Tensor([pt2.size(0)]).cuda().int()

    knn = pointops.queryandgroup(1, pt2, pt1, pt2, None, o2, o1, use_xyz=True)
    dist = torch.sum(torch.abs((knn[:,:,0:3]**2)),dim=-1)

    # idx = torch.where(dist<dist.mean())[0]
    idx = torch.where(dist<0.02)[0]

    if idx.size(0)>0:
        reduce_pt = pt1[idx]
    else:
        reduce_pt = pt1

    return reduce_pt


def clean_pts(complete, target):
    MAX = target.max(dim=0)[0]
    MIN = target.min(dim=0)[0]
    idx = torch.where( (complete[:,0]<MAX[0]) & (complete[:,1]<MAX[1]) & (complete[:,2]<MAX[2]) & (complete[:,0]>MIN[0]) & (complete[:,1]>MIN[1]) & (complete[:,2]>MIN[2]))[0]
    
    return complete[idx]




def cal_loss(c1,c2,c3,c4,c5,p1,p2,p3,p4,p5,target_points,o_tar):
# def cal_loss(c1,target_points,o0,loss_func):

    total_train_loss=0    
    batch = o_tar.size(0)

    com_1, oc1 = c1
    com_2, oc2 = c2
    com_3, oc3 = c3
    com_4, oc4 = c4
    com_5, oc5 = c5

    par_1, o1 = p1
    par_2, o2 = p2
    par_3, o3 = p3
    par_4, o4 = p4
    par_5, o5 = p5

    




    for i in range(o_tar.size(0)):
        # com = c1[i*c1.size(0)//2:(i+1)*c1.size(0)//2,:]
        # par = p1[i*p1.size(0)//2:(i+1)*p1.size(0)//2,:]
        # print(i)
        tar = target_points[:o_tar[i],:] if i==0 else target_points[o_tar[i-1]:o_tar[i],:]
        # print(oc5)
        com = com_5[:o5[i],:,:] if i==0 else com_5[o5[i-1]:o5[i],:,:]
        par = par_5[:o5[i],:] if i==0 else par_5[o5[i-1]:o5[i],:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)
        
        # com = c5[i,:,:]
        # par = p5[i,:,:]
        # tar = target_points[i*target_points.size(0)//batch:(i+1)*target_points.size(0)//batch,:]
        # com = clean_pts(com,tar)
        # print('tar',tar.size())
        # print('par',par.size())
        # print('com',com.size())
        # print('par',par.size())
        # print('tar',tar.size())

        b = reduce_pointcloud_single(tar[:par.size(0),:],par)
        c = intersection_pointcloud(com,b)
        loss1,_ = calc_cd(c.unsqueeze(0),b.unsqueeze(0))
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss1.mean()
        total_train_loss+=loss2.mean()*0.1


        com = com_4[:o4[i],:,:] if i==0 else com_4[o4[i-1]:o4[i],:,:]
        par = par_4[:o4[i],:] if i==0 else par_4[o4[i-1]:o4[i],:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)
        # com = c4[i,:,:]
        # par = p4[i,:,:]
        # tar = target_points[i*target_points.size(0)//batch:(i+1)*target_points.size(0)//batch,:]
        # com = clean_pts(com,tar)
        b = reduce_pointcloud_single(tar[:par.size(0),:],par)
        c = intersection_pointcloud(com,b)
        loss1,_ = calc_cd(c.unsqueeze(0),b.unsqueeze(0))
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss1.mean()
        total_train_loss+=loss2.mean()*0.1

        com = com_3[:o3[i],:,:] if i==0 else com_3[o3[i-1]:o3[i],:,:]
        par = par_3[:o3[i],:] if i==0 else par_3[o3[i-1]:o3[i],:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)


        # com = c3[i,:,:]
        # par = p3[i,:,:]
        # tar = target_points[i*target_points.size(0)//batch:(i+1)*target_points.size(0)//batch,:]
        # com = clean_pts(com,tar)
        b = reduce_pointcloud_single(tar[:par.size(0),:],par)
        c = intersection_pointcloud(com,b)
        loss1,_ = calc_cd(c.unsqueeze(0),b.unsqueeze(0))
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss1.mean()
        total_train_loss+=loss2.mean()*0.1


        com = com_2[:o2[i],:,:] if i==0 else com_2[o2[i-1]:o2[i],:,:]
        par = par_2[:o2[i],:] if i==0 else par_2[o2[i-1]:o2[i],:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)

        # com = c2[i,:,:]
        # par = p2[i,:,:]
        # tar = target_points[i*target_points.size(0)//batch:(i+1)*target_points.size(0)//batch,:]
        # com = clean_pts(com,tar)
        b = reduce_pointcloud_single(tar[:par.size(0),:],par)
        c = intersection_pointcloud(com,b)
        loss1,_ = calc_cd(c.unsqueeze(0),b.unsqueeze(0))
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss1.mean()
        total_train_loss+=loss2.mean()*0.1



        # com = com_1[:oc1[i],:] if i==0 else com_1[oc1[i-1]:oc1[i],:]
        par = par_1[:o1[i],:] if i==0 else par_1[o1[i-1]:o1[i],:]
        com = com_1[:o1[i],:,:] if i==0 else com_1[o1[i-1]:o1[i],:,:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)
        # com = c1[i,:,:]
        # par = p1[i,:,:]
        # tar = target_points[i*target_points.size(0)//batch:(i+1)*target_points.size(0)//batch,:]
        # com = clean_pts(com,tar)
        b = reduce_pointcloud_single(tar[:par.size(0),:],par)
        c = intersection_pointcloud(com,b)
        loss1,_ = calc_cd(c.unsqueeze(0),b.unsqueeze(0))
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss1.mean()
        total_train_loss+=loss2.mean()*0.1

    # sio.savemat('reduce_test.mat', {'c':c.cpu().detach().numpy(),'b':b.cpu().detach().numpy()})
    # total_train_loss  = loss1.mean()
    # print(total_train_loss)
    return total_train_loss




class Fold(nn.Module):
    def __init__(self, in_channel, step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

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
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        # print(fd2)
        return fd2


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




class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class TransitionUp2(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        
        p1, x1, o1 = pxo1; p2, x2, o2 = pxo2 # p1 m; p2 n
        # print('x2',x2.size())
        grouped = pointops.queryandgroup(5, p2, p1, x2, None, o2, o1, use_xyz=True)
        #  xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
        #  output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
        grouped_xyz = grouped[:,:,:3]
        grouped_feats = grouped[:,:,3:]
        dists = torch.sum(grouped_xyz**2,-1)
        dists[dists < 1e-10] = 1e-10
        weight = 1.0 / dists  # [ N,5]
        weight = weight / torch.sum(weight, dim=-1).view(grouped_feats.size(0), 1)  # [B, N, 3]
        interpolated_feats = torch.sum(grouped_feats * weight.view(grouped_feats.size(0), grouped_feats.size(1), 1), dim=1)
        # print('G',grouped_feats.size())
        # print('W',weight.size())
        # print('IP',interpolated_feats)
        # exit(0)
        if x1 is None:
            # x = pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
            x = self.linear2(interpolated_feats)
        else:
            # x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
            x = self.linear1(x1) +  self.linear2(interpolated_feats)
        return x








class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1, share_planes=8, nsample=16):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.transition = TransitionUp2(dim_feat, 128 )

        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = nn.Sequential(
                        nn.Linear(3, 64),
                        nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                        nn.Linear(64, 128))
        # MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = nn.Sequential(
                        nn.Linear(128  , 256),
                        nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                        nn.Linear(256, 128))
        #  MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = PointTransformerLayer(128, 128, share_planes, nsample)
        # SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                        nn.Linear(64, 32))
        # MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = nn.Sequential(
                        nn.Linear(128, 128),
                        nn.BatchNorm1d(128), 
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 128))
        self.short_cut = nn.Sequential(
                        nn.Linear(128, 128),
                        nn.BatchNorm1d(128) )     
        # MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = nn.Sequential(
                        nn.Linear(128 , 64),
                        nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                        nn.Linear(64, 3))
        self.linear1 = nn.Linear(128 , 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)   

        self.folding = Fold(128,128,2)     
        # MLP_CONV(in_channel=128, layer_dims=[64, 3])

        self.foldingnet = Fold(128, step = 2, hidden_dim = 256)

        
    def forward(self, pcd_prev, K_prev, pxo ):
        """
        Args:
            pcd_prev: Tensor, ( N_prev, 3)
            K_prev: Tensor, ( N_prev, 128)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        # b, _, n_prev = pcd_prev.shape
        # print('pcd_prev',pcd_prev.size())
        # print('K_prev',K_prev.size())
        # print('o',o)
        e_p, e_x, e_o = pxo
        # print(e_o)
        # exit(0)
        # o = e_o
        o = torch.tensor([torch.tensor(pcd_prev.size(0)//2), torch.tensor(pcd_prev.size(0))]).int().to(e_o.device)
        # print(o)
        feat_1 = self.mlp_1(pcd_prev)
        # feat_1 = torch.cat([feat_1,
        #                     torch.max(feat_1, 0, keepdim=True)[0].repeat((feat_1.size(0),1))],-1)
                            # feat_global.repeat(1, 1, feat_1.size(2))], 1)
        # print('feat_1',feat_1.size())
        Q = self.mlp_2(feat_1)

        # print('Q',Q.size())
        # print('K_prev',K_prev.size())
        if K_prev is None:
            H = self.relu(self.bn2(self.skip_transformer([pcd_prev, K_prev if K_prev is not None else Q, o])))
        else:
            K_prev = self.relu(self.bn1(self.linear1(K_prev)))
            identity = K_prev
            H = self.relu(self.bn2(self.skip_transformer([pcd_prev, K_prev if K_prev is not None else Q, o])))
            # x = self.bn3(self.linear3(x))
            H = H + identity

        # up_feat =self.transition([pcd_prev, K_prev,o],pxo)
        # H = up_feat

        # print(up_feat.size())
        # print(pcd_prev.size())
        # exit(0)
        # H = self.folding(up_feat)
        
        # print(H)
        
        # print('H',H)
        feat_child = self.mlp_ps(H)
        # print(feat_child.size())
        # pcd_child = (self.foldingnet(H) + pcd_prev.unsqueeze(-1)).transpose(1,2).reshape(-1, 3)


        #_______________________________________________
        # print('feat_child',feat_child.size())
        feat_child = self.ps(feat_child.transpose(1,0).contiguous().unsqueeze(0)).squeeze().transpose(1,0).contiguous()  # (B, 128, N_prev * up_factor)
        # print(feat_child)
        # print('ps_feat_child',feat_child.size())
        H_up = self.up_sampler(H.transpose(1,0).contiguous().unsqueeze(0)).squeeze().transpose(1,0).contiguous()
        # print('H_up',H_up.size())
        # K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))
        # K_curr = torch.relu(K_curr + self.short_cut(torch.cat([feat_child, H_up], 1)))
        K_curr = self.mlp_delta_feature(feat_child)
        K_curr = torch.relu(K_curr + self.short_cut(feat_child))
        # K_curr = K_curr
        
        # print('K_curr',K_curr.size())
        delta = torch.tanh(torch.tanh(self.mlp_delta(K_curr))) / self.radius ** self.i  # (B, 3, N_prev * up_factor)
        # print('delta',delta.size())
        # print('D',delta)
        pcd_child = self.up_sampler(pcd_prev.transpose(1,0).contiguous().unsqueeze(0)).squeeze().transpose(1,0).contiguous()
        # print('pcd_child',pcd_child.size())
        pcd_child = pcd_child + delta
        #_______________________________________________


        return pcd_child, K_curr



# def  mask_lowlevel(pxo):

#     p,x,o = pxo
#     # x =  torch.cat((p0, x0), 1)
#     nsample = 30
#     knn_x = pointops.queryandgroup(nsample, p, p, x, None, o, o, use_xyz=False) 
#     center_x = x.reshape(-1,1,3).repeat(1,nsample,1)
#     d_x = ((center_x - knn_x)**2).sum(-1).sum(-1) # N
#     min_d = d_x.min(dim=-1)[0].repeat([d_x.size(0)])
#     max_d = d_x.max(dim=-1)[0].repeat([d_x.size(0)])
#     d_x = (d_x-min_d)/(max_d-min_d)


#     knn_p = pointops.queryandgroup(nsample, p, p, p, None, o, o, use_xyz=False) 
#     center_p = p.reshape(-1,1,3).repeat(1,nsample,1)
#     d_p = ((center_p - knn_p)**2).sum(-1).sum(-1)
#     min_d = d_p.min(dim=-1)[0].repeat([d_p.size(0)])
#     max_d = d_p.max(dim=-1)[0].repeat([d_p.size(0)])
#     d_p = (d_p-min_d)/(max_d-min_d)


#     d = d_x + d_p
#     # print(d.size())
#     mask_p=[]
#     mask_x=[]
#     mask_o=[]
#     count=0

#     for i in range(o.size(0)):
#         p_i = p[o[i-1]:o[i],:] if i!=0  else p[:o[i],:]
#         x_i = x[o[i-1]:o[i],:] if i!=0  else x[:o[i],:]
#         d_i = d[o[i-1]:o[i]] if i!=0  else d[:o[i]]

#         mask_rate = 0.3 +0.7*torch.rand(1)
#         idx = d_i.topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
#         mask_p.append(p_i[idx])
#         mask_x.append(x_i[idx])
#         count += idx.size(0)
#         mask_o.append(count)

#     mask_p = torch.cat(mask_p)
#     mask_x = torch.cat(mask_x)
#     mask_o = torch.IntTensor(mask_o).to(p.device)

#     # print('mask_p',mask_p.size())
#     # print('mask_x',mask_x.size())
#     # print('o',o)
#     # print('mask_o',mask_o)
#     # exit(0)
#     sio.savemat('mask_test.mat', {'mask_p':mask_p.cpu().detach().numpy(),'mask_x':mask_x.cpu().detach().numpy(),'mask_o':mask_o.cpu().detach().numpy(),
#         'p':p.cpu().detach().numpy(),'x':x.cpu().detach().numpy(),'o':o.cpu().detach().numpy()})

#     return [mask_p,mask_x,mask_o]

    


def random_crop(org_coord, org_feat):

    LEN_X = torch.max(org_coord[:,0]) - torch.min(org_coord[:,0])
    LEN_Y = torch.max(org_coord[:,1]) - torch.min(org_coord[:,1])
    
    CROP_X = LEN_X/10
    CROP_Y = LEN_Y/10

    t=torch.zeros(org_coord.shape[0])
    crop_ids = []
    for i in range(10):
        center_x = torch.min(org_coord[:,0]) + np.random.randint(0,10)* CROP_X
        center_y = torch.min(org_coord[:,1]) + np.random.randint(0,10)* CROP_Y
        crop_ids.append(torch.where((org_coord[:,0]<center_x+CROP_X) & (org_coord[:,0]>center_x-CROP_X)& (org_coord[:,1]<center_y+CROP_Y) & (org_coord[:,1]>center_y-CROP_Y) )[0])
    crop_ids = torch.cat(crop_ids)
    t[crop_ids]+=1
    save_ids = torch.where(t==0)[0]
    partial_coord = org_coord[save_ids]
    partial_feat = org_feat[save_ids]

    return partial_coord, partial_feat





def  mask_lowlevel_Diff(pxo):

    p,x,o = pxo
    # x =  torch.cat((p0, x0), 1)
    nsample = 30
    knn_x = pointops.queryandgroup(nsample, p, p, x, None, o, o, use_xyz=False) 
    center_x = x.reshape(-1,1,3).repeat(1,nsample,1)
    d_x = ((center_x - knn_x)**2).sum(-1).sum(-1) # N
    min_d = d_x.min(dim=-1)[0].repeat([d_x.size(0)])
    max_d = d_x.max(dim=-1)[0].repeat([d_x.size(0)])
    d_x = (d_x-min_d)/(max_d-min_d)


    knn_p = pointops.queryandgroup(nsample, p, p, p, None, o, o, use_xyz=False) 
    center_p = p.reshape(-1,1,3).repeat(1,nsample,1)
    d_p = ((center_p - knn_p)**2).sum(-1).sum(-1)
    min_d = d_p.min(dim=-1)[0].repeat([d_p.size(0)])
    max_d = d_p.max(dim=-1)[0].repeat([d_p.size(0)])
    d_p = (d_p-min_d)/(max_d-min_d)


    d = d_x + d_p
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
        x_i = x[o[i-1]:o[i],:] if i!=0  else x[:o[i],:]
        d_i = d[o[i-1]:o[i]] if i!=0  else d[:o[i]]

        mask_rate = 0.2 + 0.7*torch.rand(1)
        idx = d_i.topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]

        # masked_coord, masked_feat = random_crop( p_i[idx], x_i[idx])
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

    # print('mask_p',mask_p.size())
    # print('mask_x',mask_x.size())
    # print('o',o)
    # print('mask_o',mask_o)
    # exit(0)
    # sio.savemat('mask_test.mat', {'mask_p':mask_p.cpu().detach().numpy(),'mask_x':mask_x.cpu().detach().numpy(),'mask_o':mask_o.cpu().detach().numpy(),
    #     'p':p.cpu().detach().numpy(),'x':x.cpu().detach().numpy(),'o':o.cpu().detach().numpy()})

    return [mask_p,mask_x,mask_o], [gt_p, gt_x, gt_o]



# def  mask_lowlevel(pxo):

#     p,x,o = pxo
#     # x =  torch.cat((p0, x0), 1)
#     nsample = 30
#     knn_x = pointops.queryandgroup(nsample, p, p, x, None, o, o, use_xyz=False) 
#     center_x = x.reshape(-1,1,3).repeat(1,nsample,1)
#     d_x = ((center_x - knn_x)**2).sum(-1).sum(-1) # N
#     min_d = d_x.min(dim=-1)[0].repeat([d_x.size(0)])
#     max_d = d_x.max(dim=-1)[0].repeat([d_x.size(0)])
#     d_x = (d_x-min_d)/(max_d-min_d)


#     knn_p = pointops.queryandgroup(nsample, p, p, p, None, o, o, use_xyz=False) 
#     center_p = p.reshape(-1,1,3).repeat(1,nsample,1)
#     d_p = ((center_p - knn_p)**2).sum(-1).sum(-1)
#     min_d = d_p.min(dim=-1)[0].repeat([d_p.size(0)])
#     max_d = d_p.max(dim=-1)[0].repeat([d_p.size(0)])
#     d_p = (d_p-min_d)/(max_d-min_d)


#     d = d_x + d_p
#     # print(d.size())
#     mask_p=[]
#     mask_x=[]
#     mask_o=[]
#     count=0

#     mask_p_05=[]
#     mask_x_05=[]
#     mask_o_05=[]
#     mask_o_05=[]
#     count_05=0


#     mask_p_07=[]
#     mask_x_07=[]
#     mask_o_07=[]
#     mask_o_07=[]
#     count_07=0


#     mask_p_09=[]
#     mask_x_09=[]
#     mask_o_09=[]
#     mask_o_09=[]
#     count_09=0

#     for i in range(o.size(0)):
#         p_i = p[o[i-1]:o[i],:] if i!=0  else p[:o[i],:]
#         x_i = x[o[i-1]:o[i],:] if i!=0  else x[:o[i],:]
#         d_i = d[o[i-1]:o[i]] if i!=0  else d[:o[i]]

#         mask_rate = 0.3
#         idx = d_i.topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
#         mask_p.append(p_i[idx])
#         mask_x.append(x_i[idx])
#         count += idx.size(0)
#         mask_o.append(count)

#         mask_rate = 0.5
#         idx = d_i.topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
#         mask_p_05.append(p_i[idx])
#         mask_x_05.append(x_i[idx])
#         count_05 += idx.size(0)
#         mask_o_05.append(count_05)

#         mask_rate = 0.7
#         idx = d_i.topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
#         mask_p_07.append(p_i[idx])
#         mask_x_07.append(x_i[idx])
#         count_07 += idx.size(0)
#         mask_o_07.append(count_07)

#         mask_rate = 0.9
#         idx = d_i.topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
#         mask_p_09.append(p_i[idx])
#         mask_x_09.append(x_i[idx])
#         count_09 += idx.size(0)
#         mask_o_09.append(count_09)


#     mask_p = torch.cat(mask_p)
#     mask_p_05 = torch.cat(mask_p_05)
#     mask_p_07 = torch.cat(mask_p_07)
#     mask_p_09 = torch.cat(mask_p_09)

#     mask_x = torch.cat(mask_x)
#     mask_x_05 = torch.cat(mask_x_05)
#     mask_x_07 = torch.cat(mask_x_07)
#     mask_x_09 = torch.cat(mask_x_09)



#     mask_o = torch.IntTensor(mask_o).to(p.device)
#     mask_o_05 = torch.IntTensor(mask_o_05).to(p.device)
#     mask_o_07 = torch.IntTensor(mask_o_07).to(p.device)
#     mask_o_09 = torch.IntTensor(mask_o_09).to(p.device)

#     sio.savemat('mask_test_sample.mat', {'mask_p':mask_p.cpu().detach().numpy(),'mask_x':mask_x.cpu().detach().numpy(),'mask_o':mask_o.cpu().detach().numpy(),
#         'p':p.cpu().detach().numpy(),'x':x.cpu().detach().numpy(),'o':o.cpu().detach().numpy(),
#         'mask_p_05':mask_p_05.cpu().detach().numpy(),'mask_x_05':mask_x_05.cpu().detach().numpy(),'mask_o_05':mask_o_05.cpu().detach().numpy(),
#         'mask_p_07':mask_p_07.cpu().detach().numpy(),'mask_x_07':mask_x_07.cpu().detach().numpy(),'mask_o_07':mask_o_07.cpu().detach().numpy(),
#         'mask_p_09':mask_p_09.cpu().detach().numpy(),'mask_x_09':mask_x_09.cpu().detach().numpy(),'mask_o_09':mask_o_09.cpu().detach().numpy()})
#     print('saved')
#     exit(0)

#     return [mask_p,mask_x,mask_o]





class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]



def pts_concate(pxo1,pxo2):
    p1,x1,o1 = pxo1
    p2,x2,o2 = pxo2
    b = o1.size(0)

    pts = []
    feats = []
    for i in range(b):
        if i == 0:
            pts1 = p1[0:o1[i],:]
            feat1 = x1[0:o1[i],:]
            pts2 = p2[0:o2[i],:]
            feat2 = x2[0:o2[i],:]          
        else:
            pts1 = p1[o1[i-1]:o1[i],:]
            feat1 = x1[o1[i-1]:o1[i],:]
            pts2 = p2[o2[i-1]:o2[i],:]
            feat2 = x2[o2[i-1]:o2[i],:]  
        pts.append(torch.cat([pts1,pts2],dim=0))
        feats.append(torch.cat([feat1,feat2],dim=0))

    # p1 = p1.reshape(b,p1.size(0)//b,-1) # b,N,3
    # p2 = p2.reshape(b,p2.size(0)//b,-1)
    # # print(p2.size())
    # p = torch.cat([p1,p2],dim=1).reshape(-1,3)
    # x1 = x1.reshape(b,x1.size(0)//b,-1) 
    # x2 = x2.reshape(b,x2.size(0)//b,-1) 
    # x = torch.cat([x1,x2],dim=1).reshape(-1,x1.size(-1))
    o=o1+o2
    return [torch.cat(pts),torch.cat(feats),o]



def pts_reduce(x,o,xo):

    b = o.size(0)

    feats = []
    for i in range(b):
        if i == 0:
            feat = x[0:o[i],:]
        else:
            feat = x[xo[i-1]:xo[i-1]+o[i]-o[i-1],:]
        # pts.append(torch.cat([pts1,pts2],dim=0))
        feats.append(feat)
    return torch.cat(feats)
    # p1 = p1.reshape(b,p1.size(0)//b,-1) # b,N,3
    # p2 = p2.reshape(b,p2.size(0)//b,-1)
    # # print(p2.size())
    # p = torch.cat([p1,p2],dim=1).reshape(-1,3)
    # x1 = x1.reshape(b,x1.size(0)//b,-1) 
    # x2 = x2.reshape(b,x2.size(0)//b,-1) 
    # x = torch.cat([x1,x2],dim=1).reshape(-1,x1.size(-1))
    # o=o1+o2
    # return [torch.cat(pts),torch.cat(feats),o]





# def pts_concate(pxo1,pxo2):
#     p1,x1,o1 = pxo1
#     p2,x2,o2 = pxo2
#     b = o1.size(0)
#     p1 = p1.reshape(b,p1.size(0)//b,-1) # b,N,3
#     p2 = p2.reshape(b,p2.size(0)//b,-1)
#     # print(p2.size())
#     p = torch.cat([p1,p2],dim=1).reshape(-1,3)
#     x1 = x1.reshape(b,x1.size(0)//b,-1) 
#     x2 = x2.reshape(b,x2.size(0)//b,-1) 
#     x = torch.cat([x1,x2],dim=1).reshape(-1,x1.size(-1))
#     o=o1+o2
#     return [p,x,o]

def pts_shape_reset(com_pts,off):
    # com_pts [N,3,4]
    pts = []
    for i in range(off.size(0)):
        if i==0:
            pt = com_pts[: off[i],:,:]
        else:
            pt = com_pts[off[i-1]: off[i],:,:]
        pts.append(pt.transpose(2,1).contiguous().reshape(-1, 3))
    return torch.cat(pts,dim=0)


def pts_agg_2batch(pxo1,pxo2):
    p1,x1,o1 = pxo1
    p2,x2,o2 = pxo2
    b = o1.size(0)

    pts = torch.cat([p1,p2], dim=0)
    feats = torch.cat([x1,x2], dim=0)
    o = torch.cat([o1,o2+o1[-1]])
    return [pts, feats, o]



def pts_div_2batch(pxo):
    p,x,o = pxo
    b = o.size(0)
    mid = b//2 
    p1 = p[:o[mid-1],:]
    x1 = x[:o[mid-1],:]
    o1 = o[:mid]

    p2 = p[o[mid-1]:,:]
    x2 = x[o[mid-1]:,:]
    o2 = o[mid:]-o[mid-1]

    return [p1, x1, o1], [p2, x2, o2] 

def get_partial_global(x,o):
    g = []
    for i in range(o.size(0)):
        x_i = x[:o[i],:] if i==0 else x[o[i-1]:o[i],:] 
        g.append(torch.max(x_i,dim=0)[0].reshape(1,-1))
    return torch.cat(g)




class PointTransformer_model(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)


    def forward(self, pxto):
        p0, x0, o0 = pxto  # (n, 3), (n, c), (b)

        # rgb_0 = x0
        # masked_0, masked_RGB0, masked_o0 = mxto # masked points

        ###  branch completion encoder -------------------
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
       

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        return [p1,x1,o1],[p2,x2,o2], [p3,x3,o3], [p4,x4,o4], [p5,x5,o5] 







class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, moving_average_decay = 0.99):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.target_ema_updater = EMA(moving_average_decay)
        self.target_encoder = None
        self.PT_model = PointTransformer_model(PointTransformerBlock, [2, 3, 4, 6, 3])


        # self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        # self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        # self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        # self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        # self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        # self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        # self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        # self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        # self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        # self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        # self.cls_pt = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        
        # self.down_sample_1 = TransitionDown(3, 3 * block.expansion, stride[0], nsample[0])
        # self.com5 = SPD(dim_feat=planes[4], up_factor=2, i=0, radius=2, share_planes=8, nsample=nsample[4])
        # self.com4 = SPD(dim_feat=planes[3], up_factor=2, i=1, radius=2, share_planes=8, nsample=nsample[3])
        # self.com3 = SPD(dim_feat=planes[2], up_factor=2, i=2, radius=2, share_planes=8, nsample=nsample[2])
        # self.com2 = SPD(dim_feat=planes[1], up_factor=2, i=3, radius=2, share_planes=8, nsample=nsample[1])
        # self.com1 = SPD(dim_feat=planes[0], up_factor=2, i=4, radius=2, share_planes=8, nsample=nsample[0])

        # self.transition5 = TransitionUp2(planes[4], 128 )
        # self.transition4 = TransitionUp2(planes[3], 128 )
        # self.transition3 = TransitionUp2(planes[2], 128 )
        # self.transition2 = TransitionUp2(planes[1], 128 )
        # self.transition1 = TransitionUp2(planes[0], 128 )
        self.foldingnet1 = Fold(planes[0]+3, step = 2, hidden_dim = 256)
        self.foldingnet2 = Fold(planes[1]+3, step = stride[1]//2, hidden_dim = 256)
        self.foldingnet3 = Fold(planes[2]+3, step = stride[2]//2, hidden_dim = 256)
        self.foldingnet4 = Fold(planes[3]+3, step = stride[3]//2, hidden_dim = 256)
        self.foldingnet5 = Fold(planes[4]+3, step = stride[4]//2, hidden_dim = 256)

        self.m_foldingnet1 = Fold(planes[0]+3, step = 2, hidden_dim = 256)
        self.m_foldingnet2 = Fold(planes[1]+3, step = stride[1]//2, hidden_dim = 256)
        self.m_foldingnet3 = Fold(planes[2]+3, step = stride[2]//2, hidden_dim = 256)
        self.m_foldingnet4 = Fold(planes[3]+3, step = stride[3]//2, hidden_dim = 256)
        self.m_foldingnet5 = Fold(planes[4]+3, step = stride[4]//2, hidden_dim = 256)
        # self.pool = nn.MaxPool1d(nsample)
        # self.pos_embedding = PositionEmbeddingCoordsSine(
        #     d_pos=10, pos_type="fourier", normalize=True)
        # self.pos_embed = nn.Sequential(nn.Linear(10, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Linear(32, planes[0]))

        # trans = []
        # trans.append(block(planes[0], planes[0], share_planes, nsample=64))
        # trans.append(block(planes[0], planes[0], share_planes, nsample=32))
        # trans.append(block(planes[0], planes[0], share_planes, nsample=16))
        # self.trans = nn.Sequential(*trans)
        # self.fc_rgb = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 3))
        


    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)


    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.PT_model)
        set_requires_grad(target_encoder, False)
        return target_encoder


    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        # assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.PT_model)




    def forward(self, pxto, m_pxo):

        pxto, gt_pxo = mask_lowlevel_Diff(m_pxo)


        pxo1,pxo2,pxo3,pxo4,pxo5 = self.PT_model(pxto)
        p1,x1,o1 = pxo1
        p2,x2,o2 = pxo2
        p3,x3,o3 = pxo3
        p4,x4,o4 = pxo4
        p5,x5,o5 = pxo5


        # p0, x0, o0 = pxto  # (n, 3), (n, c), (b)

        # rgb_0 = x0
        # # masked_0, masked_RGB0, masked_o0 = mxto # masked points

        # ###  branch completion encoder -------------------
        # x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        # p1, x1, o1 = self.enc1([p0, x0, o0])
        # p2, x2, o2 = self.enc2([p1, x1, o1])
        # p3, x3, o3 = self.enc3([p2, x2, o2])
        # p4, x4, o4 = self.enc4([p3, x3, o3])
        # p5, x5, o5 = self.enc5([p4, x4, o4])
       

        ## ------------------
        x5_ = torch.cat([
            # torch.max(x5, dim=0)[0].unsqueeze(0).expand(p5.size(0), -1),
            x5,
            p5], dim= -1)  

        c5 = (self.foldingnet5(x5_) + p5.unsqueeze(-1))
        oc5 = o5*4

        oc4 = o4*4
        x4_ = torch.cat([
            # torch.max(x4, dim=0)[0].unsqueeze(0).expand(p4.size(0), -1),
            x4,
            p4], dim= -1)  
        c4 = (self.foldingnet4(x4_) + p4.unsqueeze(-1))


        x3_ = torch.cat([
            # torch.max(x3, dim=0)[0].unsqueeze(0).expand(p3.size(0), -1),
            x3,
            p3], dim= -1)          
        c3 = (self.foldingnet3(x3_)+ p3.unsqueeze(-1))
        # .transpose(1,2).reshape(-1, 3)
        oc3 = o3*4

        
        x2_ = torch.cat([
            x2,
            p2], dim= -1)  
        c2 = (self.foldingnet2(x2_) + p2.unsqueeze(-1))

        oc2=o2*4

        x1_ = torch.cat([
            # torch.max(x1, dim=0)[0].unsqueeze(0).expand(p1.size(0), -1),
            x1,
            p1], dim= -1)  
        c1 = self.foldingnet1(x1_) + p1.unsqueeze(-1)

        oc1=o1*4



        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            m_pxo1,m_pxo2,m_pxo3,m_pxo4,m_pxo5 = target_encoder(gt_pxo)
            m_p1,m_x1,m_o1 = m_pxo1
            m_p2,m_x2,m_o2 = m_pxo2
            m_p3,m_x3,m_o3 = m_pxo3
            m_p4,m_x4,m_o4 = m_pxo4
            m_p5,m_x5,m_o5 = m_pxo5

            ct_par_x1 = pointops.interpolation(m_p1, p1, m_x1, m_o1, o1).detach_()
            ct_par_x2 = pointops.interpolation(m_p2, p2, m_x2, m_o2, o2).detach_()
            ct_par_x3 = pointops.interpolation(m_p3, p3, m_x3, m_o3, o3).detach_()
            ct_par_x4 = pointops.interpolation(m_p4, p4, m_x4, m_o4, o4).detach_()
            ct_par_x5 = pointops.interpolation(m_p5, p5, m_x5, m_o5, o5).detach_()


        return [c1,oc1],[c2,oc2],[c3,oc3],[c4,oc4],[c5,oc5],[p1,o1],[p2,o2],[p3,o3],[p4,o4],[p5,o5],gt_pxo,\
            [ct_par_x1,x1,o1],\
            [ct_par_x2,x2,o2],\
            [ct_par_x3,x3,o3],\
            [ct_par_x4,x4,o4],\
            [ct_par_x5,x5,o5]   





        # return [c1,oc1],[c2,oc2],[c3,oc3],[c4,oc4],[c5,oc5],[p1,o1],[p2,o2],[p3,o3],[p4,o4],[p5,o5], gt_pxo
        # ,\
        #     [ct_par_x1,x1,o1],\
        #     [ct_par_x2,x2,o2],\
        #     [ct_par_x3,x3,o3],\
        #     [ct_par_x4,x4,o4],\
        #     [ct_par_x5,x5,o5]



def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model
