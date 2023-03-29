from shutil import copy2
import torch
import torch.nn as nn

from lib.pointops.functions import pointops
from model.pointtransformer.model_utils import *
from model.pointtransformer.SPD_refine import *
import numpy as np
# from lib.chamfer_dist import ChamferDistanceL1

def cal_loss_scene(predict,target,loss_func):

        LEN_X = torch.max(target[:,0]) - torch.min(target[:,0])
        LEN_Y = torch.max(target[:,1]) - torch.min(target[:,1])
        
        CROP_X = LEN_X/5
        CROP_Y = LEN_Y/5
        # print(CROP_X)
        # crop_ids = []
        losses = 0
        for i in range(25):
            # crop_center = org_coord[np.random.randint(0,org_coord.shape[0]),:] 
            center_x = torch.min(target[:,0]) + (i//5)* CROP_X
            center_y = torch.min(target[:,1]) + (i%5)* CROP_Y
            # print(center_x)
            tar_ids=torch.where((target[:,0]<center_x+CROP_X) & (target[:,0]>center_x-CROP_X)& (target[:,1]<center_y+CROP_Y) & (target[:,1]>center_y-CROP_Y) )[0]
            pre_ids=torch.where((predict[:,0]<center_x+CROP_X) & (predict[:,0]>center_x-CROP_X)& (predict[:,1]<center_y+CROP_Y) & (predict[:,1]>center_y-CROP_Y) )[0]
            if pre_ids.shape[0]>10 and  tar_ids.shape[0]>10 :
                a = predict[pre_ids].unsqueeze(0)
                b = target[tar_ids].unsqueeze(0)
                losses+=(loss_func(predict[pre_ids].unsqueeze(0), target[tar_ids].unsqueeze(0)))
        # crop_ids = np.concatenate(crop_ids)
        # print(losses)
        # exit(0)
        # losses = torch.cat(losses,dim=0)
        return losses
        # t[crop_ids]+=1
        # save_ids = torch.where(t==0)[0]
        # partial_coord = org_coord[save_ids]
        # partial_feat = org_feat[save_ids]
        # partial_label = org_label[save_ids]

def cal_loss(c5,c4,c3,c2,c1,target_points,o0,loss_func):
# def cal_loss(c1,target_points,o0,loss_func):
        target = target_points.reshape(o0.size(0),target_points.size(0)//o0.size(0),3)
        # target= target_points
        # for i in range(o0.size(0)):
        #     tar_points = target_points[:o0[i]]
        # print('tar',target.size())
        # c5 = c5.reshape(o0.size(0),c5.size(0)//o0.size(0),3)
        # c4 = c4.reshape(o0.size(0),c4.size(0)//o0.size(0),3)
        # c3 = c3.reshape(o0.size(0),c3.size(0)//o0.size(0),3)
        # c2 = c2.reshape(o0.size(0),c2.size(0)//o0.size(0),3)
        # c1 = c1.reshape(o0.size(0),c1.size(0)//o0.size(0),3)

        # # print(p5.size())
        # p5 = p5.reshape(o0.size(0),p5.size(0)//o0.size(0),3)
        # p4 = p4.reshape(o0.size(0),p4.size(0)//o0.size(0),3)
        # p3 = p3.reshape(o0.size(0),p3.size(0)//o0.size(0),3)
        # p2 = p2.reshape(o0.size(0),p2.size(0)//o0.size(0),3)
        # p1 = p1.reshape(o0.size(0),p1.size(0)//o0.size(0),3)   
        # c5 = torch.cat([c5,p5], dim=1)
        # c4 = torch.cat([c4,p4], dim=1)
        # c3 = torch.cat([c3,p3], dim=1)
        # c2 = torch.cat([c2,p2], dim=1)
        # c1 = torch.cat([c1,p1], dim=1)
        # print(c1.size())

        # loss5, _ = calc_cd(c5, target[:c5.size(1),:])
        # loss4, _ = calc_cd(c4, target[:c4.size(1),:])
        # loss3, _ = calc_cd(c3, target[:c3.size(1),:])
        # loss2, _ = calc_cd(c2, target[:c2.size(1),:])
        loss1 = loss_func(c1, target)
        loss2 = loss_func(c2, target)
        loss3 = loss_func(c3, target)
        loss4 = loss_func(c4, target)
        loss5 = loss_func(c5, target)
        # loss5, _ = calc_cd(torch.cat((p5,c5),dim=1), target)
        # loss4, _ = calc_cd(torch.cat((p4,c4),dim=1), target)
        # loss3, _ = calc_cd(torch.cat((p3,c3),dim=1), target)
        # loss2, _ = calc_cd(torch.cat((p2,c2),dim=1), target)
        # loss1, _ = calc_cd(torch.cat((p1,c1),dim=1), target)
        # print(loss5.mean().size())
        total_train_loss = loss5.mean() + loss4.mean() +loss3.mean() +loss2.mean() +loss1.mean()
        # total_train_loss = loss1.mean()
        # print(total_train_loss)-----------------------------------------
        # print('c1',c1.size())
        # loss1 = cal_loss_scene(c1[0,:,:],target[0,:,:],loss_func) + cal_loss_scene(c1[1,:,:],target[1,:,:],loss_func)
        # loss2 = cal_loss_scene(c2[0,:,:],target[0,:,:],loss_func) + cal_loss_scene(c2[1,:,:],target[1,:,:],loss_func)
        # loss3 = cal_loss_scene(c3[0,:,:],target[0,:,:],loss_func) + cal_loss_scene(c3[1,:,:],target[1,:,:],loss_func)
        # loss4 = cal_loss_scene(c4[0,:,:],target[0,:,:],loss_func) + cal_loss_scene(c4[1,:,:],target[1,:,:],loss_func)
        # loss5 = cal_loss_scene(c5[0,:,:],target[0,:,:],loss_func) + cal_loss_scene(c5[1,:,:],target[1,:,:],loss_func)
        # total_train_loss = loss5 + loss4 +loss3 +loss2 +loss1





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





# class Folding(nn.Module):
#     def __init__(self, input_size, output_size, step_ratio, num_models=1):
#         super(Folding, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.step_ratio = step_ratio
#         self.num_models = num_models

#         self.conv = nn.Linear(input_size  + 2, output_size)

#         sqrted = int(math.sqrt(step_ratio)) + 1
#         for i in range(1, sqrted + 1).__reversed__():
#             if (step_ratio % i) == 0:
#                 num_x = i
#                 num_y = step_ratio // i
#                 break

#         grid_x = torch.linspace(-0.2, 0.2, steps=num_x)
#         grid_y = torch.linspace(-0.2, 0.2, steps=num_y)

#         x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
#         self.grid = torch.stack([x, y], dim=-1).view(-1, 2)  # (2, 2)

#     def forward(self, point_feat):
#         num_points,num_features = point_feat.size()
#         # point_feat = point_feat.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, self.step_ratio, 1).view(
#         #     batch_size,
#         #     -1, num_features).transpose(1, 2).contiguous()
#         # global_feat = global_feat.unsqueeze(2).repeat(1, 1, num_points * self.step_ratio).repeat(self.num_models, 1, 1)
#         # print(point_feat.size())
#         # print(self.grid.size())
#         # exit(0)
#         grid_feat = self.grid.repeat(num_points//2, 1).contiguous().cuda()
#         # print(grid_feat.size())
#         # exit(0)
#         features = torch.cat([ point_feat, grid_feat], axis=1)
#         features = F.relu(self.conv(features))
#         return features

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



# class SkipTransformerLayer(nn.Module):
#     def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
#         super().__init__()
#         self.mid_planes = mid_planes = out_planes // 1
#         self.out_planes = out_planes
#         self.share_planes = share_planes
#         self.nsample = nsample
#         self.linear_q = nn.Linear(in_planes, mid_planes)
#         self.linear_k = nn.Linear(in_planes, mid_planes)
#         self.linear_v = nn.Linear(in_planes*2, out_planes)
#         self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
#         self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
#                                     nn.Linear(mid_planes, mid_planes // share_planes),
#                                     nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
#                                     nn.Linear(out_planes // share_planes, out_planes // share_planes))
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, pxo, pxo2) -> torch.Tensor:
#         p, x, o = pxo  # (n, 3), (n, c), (b)
#         p2, x2, xo2 = pxo2
#         v= torch.cat([x,x2],dim=-1)
#         x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(v)  # (n, c)
#         x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
#         x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
#         p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
#         for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
#         w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
#         for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
#         w = self.softmax(w)  # (n, nsample, c)
#         n, nsample, c = x_v.shape; s = self.share_planes
#         x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
#         return x



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





# class SPDUp(nn.Module):
#     def __init__(self, in_planes, out_planes=None):
#         super().__init__()
#         if out_planes is None:
#             self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
#             self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
#         else:
#             self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
#             self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
#     def forward(self, pxo1, pxo2=None):
#         if pxo2 is None:
#             _, x, o = pxo1  # (n, 3), (n, c), (b)
#             x_tmp = []
#             for i in range(o.shape[0]):
#                 if i == 0:
#                     s_i, e_i, cnt = 0, o[0], o[0]
#                 else:
#                     s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
#                 x_b = x[s_i:e_i, :]
#                 x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
#                 x_tmp.append(x_b)
#             x = torch.cat(x_tmp, 0)
#             x = self.linear1(x)
#         else:
#             p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
#             x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
#         return x




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







class SPD2(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1, share_planes=8, nsample=16):
        """Snowflake Point Deconvolution"""
        super(SPD2, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128, layer_dims=[256, 128])

        self.skip_transformer = PointTransformerLayer(128, 128, share_planes, nsample)
        # SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])
        self.linear1 = nn.Linear(128 , 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)        
        # MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, K_prev, o ):
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
        o[0] = pcd_prev.size(0)//2
        o[1] = pcd_prev.size(0)
        feat_1 = self.mlp_1(pcd_prev.transpose(1,0).contiguous().unsqueeze(0))
        # feat_1 = torch.cat([feat_1,
        #                     torch.max(feat_1, 0, keepdim=True)[0].repeat((feat_1.size(0),1))],-1)
                            # feat_global.repeat(1, 1, feat_1.size(2))], 1)
        # print('feat_1',feat_1.size())
        Q = self.mlp_2(feat_1)
        # print('Q',Q.size())
        # print('K_prev',K_prev.size())
        if K_prev is None:
            Q = Q.squeeze().transpose(1,0).contiguous()
            # pcd_prev =  pcd_prev
            H = self.relu(self.bn2(self.skip_transformer([pcd_prev, K_prev if K_prev is not None else Q, o])))
            H = H.transpose(1,0).contiguous().unsqueeze(0)
        else:
            # K_prev = K_prev.squeeze().transpose(1,0).contiguous()
            K_prev = self.relu(self.bn1(self.linear1(K_prev)))
            identity = K_prev
            H = self.relu(self.bn2(self.skip_transformer([pcd_prev, K_prev if K_prev is not None else Q, o])))
            # x = self.bn3(self.linear3(x))
            H = H + identity
            H = H.transpose(1,0).contiguous().unsqueeze(0)
        
        
        # print('H',H.size())
        # exit(0)
        feat_child = self.mlp_ps(H)
        # print('feat_child',feat_child.size())
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        # print('ps_feat_child',feat_child.size())
        H_up = self.up_sampler(H)
        # print('H_up',H_up.size())
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))
        # K_curr = torch.relu(K_curr + self.short_cut(torch.cat([feat_child, H_up], 1)))
        # K_curr = K_curr
        # print(K_curr)
        # print('K_curr',K_curr.size())
        delta = torch.tanh(self.mlp_delta(K_curr)) / self.radius ** self.i  # (B, 3, N_prev * up_factor)
        # print('delta',delta.size())
        # print(delta)
        pcd_child = self.up_sampler(pcd_prev.transpose(1,0).contiguous().unsqueeze(0)).squeeze().transpose(1,0).contiguous()
        # print('pcd_child',pcd_child.size())
        pcd_child = pcd_child + delta.squeeze().transpose(1,0).contiguous()

        return pcd_child, K_curr.squeeze().transpose(1,0).contiguous()









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




def  add_mask_lowlevel(pxo,t,epoch):

    p,x,o = pxo
    # x =  torch.cat((p0, x0), 1)
    nsample = 30
    knn_x = pointops.queryandgroup(nsample, p, p, x, None, o, o, use_xyz=False) 
    center_x = x.reshape(-1,1,3).repeat(1,nsample,1)
    d_x = ((center_x - knn_x)**2).sum(-1).sum(-1) # N
    # min_d = d_x.min(dim=1)[0].repeat([num_points])
    # max_d = d_x.max(dim=1)[0].repeat([num_points])
    # d_x = (d_x-min_d)/(max_d-min_d)


    # knn_p = pointops.queryandgroup(nsample, p, p, p, None, o, o, use_xyz=False) 
    # center_p = p.reshape(-1,1,3).repeat(1,nsample,1)
    # d_p = ((center_p - knn_p)**2).sum(-1).sum(-1)



    d = d_x
    # print(d.size())
    added_p=[]
    added_x=[]
    added_t = []
    added_o=[]
    count=0

    for i in range(o.size(0)):
        p_i = p[o[i-1]:o[i],:] if i!=0  else p[:o[i],:]
        x_i = x[o[i-1]:o[i],:] if i!=0  else x[:o[i],:]
        d_i = d[o[i-1]:o[i]] if i!=0  else d[:o[i]]
        t_i = t[o[i-1]:o[i]] if i!=0  else t[:o[i]]

        # mask_rate = 0.3 +0.7*((epoch//30)+1)*0.2
        mask_rate = 0.5*torch.rand(1)
        idx = (-1*d_i).topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
        added_p.append(torch.cat([p_i,p_i[idx]]))
        added_x.append(torch.cat([x_i,x_i[idx]]))
        added_t.append(torch.cat([t_i,t_i[idx]]))
        count += (idx.size(0) + p_i.size(0))
        added_o.append(count)

    added_p = torch.cat(added_p)
    added_x = torch.cat(added_x)
    added_t = torch.cat(added_t)
    added_o = torch.IntTensor(added_o).to(p.device)

    # print('mask_p',mask_p.size())
    # print('mask_x',mask_x.size())
    # print('o',o)
    # print('mask_o',mask_o)
    # exit(0)
    # sio.savemat('mask_test.mat', {'mask_p':mask_p.cpu().detach().numpy(),'mask_x':mask_x.cpu().detach().numpy(),'mask_o':mask_o.cpu().detach().numpy(),
    #     'p':p.cpu().detach().numpy(),'x':x.cpu().detach().numpy(),'o':o.cpu().detach().numpy()})


    return [added_p,added_x,added_o], added_t

def split_pointcloud(x,off,org_off):
    split_x=[]
    for i in range(off.size(0)):
        x_i = x[off[i-1]:off[i],:] if i!=0  else x[:off[i],:]
        org_off_i = org_off[i] - org_off[i-1] if i!=0 else org_off[i] 
        split_x.append(x_i[:org_off_i,:])
    split_x = torch.cat(split_x)

    return split_x


def  add_mask_lowlevel_FORTEST(pxo, epoch):

    p,x,o = pxo
    # x =  torch.cat((p0, x0), 1)
    nsample = 30
    knn_x = pointops.queryandgroup(nsample, p, p, x, None, o, o, use_xyz=False) 
    center_x = x.reshape(-1,1,3).repeat(1,nsample,1)
    d_x = ((center_x - knn_x)**2).sum(-1).sum(-1) # N

    d = d_x
    # print(d.size())
    added_p=[]
    added_x=[]
    added_o=[]
    count=0

    for i in range(o.size(0)):
        p_i = p[o[i-1]:o[i],:] if i!=0  else p[:o[i],:]
        x_i = x[o[i-1]:o[i],:] if i!=0  else x[:o[i],:]
        d_i = d[o[i-1]:o[i]] if i!=0  else d[:o[i]]

        # mask_rate = 0.3 +0.7*((epoch//30)+1)*0.2
        mask_rate = 0.5*torch.rand(1)
        idx = (-1*d_i).topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
        added_p.append(torch.cat([p_i,p_i[idx]]))
        added_x.append(torch.cat([x_i,x_i[idx]]))
        count += (idx.size(0) + p_i.size(0))
        added_o.append(count)

    added_p = torch.cat(added_p)
    added_x = torch.cat(added_x)
    added_o = torch.IntTensor(added_o).to(p.device)

    return [added_p,added_x,added_o]

class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
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
        self.cls_pt = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))





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

    def forward(self, pxto, tar=None, epoch=-1, training=True):

        if epoch >-1:
            pxto, tar = add_mask_lowlevel(pxto, tar, epoch)

        # if training == False:
        #     pxto_m1 = add_mask_lowlevel_FORTEST(pxto, epoch)
            # pxto_m2 = add_mask_lowlevel_FORTEST(pxto, epoch)

        pxo1,pxo2,pxo3,pxo4,pxo5 = self.PT_model(pxto)
        p1,x1,o1 = pxo1
        x = self.cls_pt(x1)


            
        if epoch >-1:
            return  x, tar
        else:
            return  x


def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model
