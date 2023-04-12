from shutil import copy2
from ssl import PROTOCOL_TLSv1_1
from tkinter import S
import torch
import torch.nn as nn

from lib.pointops.functions import pointops
from model.pointtransformer.model_utils import *
import numpy as np
import scipy.io as sio

from functools import wraps
import copy

'''
The Singleton pattern is a design pattern that restricts the instantiation of a class to one object.
 It ensures that only one instance of a class exists and provides a global point of access to that instance.
'''
def singleton(cache_key):
    def inner_fn(fn):#  The inner function takes another function as an argument and returns a wrapper function 
        @wraps(fn)
        '''
        The wrapper function checks if an instance of the class already exists in the cache using the cache key.
         If it does, it returns that instance. If it doesn’t, it creates a new instance using the original function and caches it using the cache key
        '''
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

# returns the device of the first parameter of the input module
def get_module_device(module):
    return next(module.parameters()).device


# sets the requires_grad attribute of all parameters in the input model to val
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val



# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    # Updates the average with old and new values.
    def update_average(self, old, new):
        if old is None: 
            return new
        return old * self.beta + (1 - self.beta) * new


# updates the moving average of the model parameters using EMA
def update_moving_average(ema_updater, ma_model, current_model):
    '''
    ema_updater: An instance of the EMA() class.
    ma_model: The model whose moving average needs to be updated.
    current_model: The current model parameters.
    '''
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)




# Calculates completion loss for MM-3DScene
def cal_loss(c1,c2,c3,c4,c5,target_points,o_tar):
'''
c1,c2,c3,c4,c5: Completion scenes with different sampling rates of input scene
target_points: Coordinates of target scenes 
o_tar: Offset of target scenes
'''
    total_train_loss=0    
    batch = o_tar.size(0)

    com_1, o1 = c1 # (n1,3,nsample), (b)
    com_2, o2 = c2 # (n2,3,nsample), (b)
    com_3, o3 = c3 # (n2,3,nsample), (b)
    com_4, o4 = c4 # (n4,3,nsample), (b)
    com_5, o5 = c5 # (n5,3,nsample), (b)

    for i in range(o_tar.size(0)):
        tar = target_points[:o_tar[i],:] if i==0 else target_points[o_tar[i-1]:o_tar[i],:]

        com = com_5[:o5[i],:,:] if i==0 else com_5[o5[i-1]:o5[i],:,:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss2.mean()


        com = com_4[:o4[i],:,:] if i==0 else com_4[o4[i-1]:o4[i],:,:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss2.mean()

        com = com_3[:o3[i],:,:] if i==0 else com_3[o3[i-1]:o3[i],:,:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss2.mean()


        com = com_2[:o2[i],:,:] if i==0 else com_2[o2[i-1]:o2[i],:,:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss2.mean()

        com = com_1[:o1[i],:,:] if i==0 else com_1[o1[i-1]:o1[i],:,:]
        com = com.transpose(1,2).contiguous().reshape(-1, 3)
        loss2,_ = calc_cd(com.unsqueeze(0),tar.unsqueeze(0))
        total_train_loss+=loss2.mean()

    return total_train_loss












def Informative_Perserved_Mask(pxo, completion_gap = 0.1):
    p,x,o = pxo  # p: coordinates(n, 3), x: rgb  (n, 3), offset: (b)
    nsample = 30

    # calculate the local difference of coordinates
    knn_x = pointops.queryandgroup(nsample, p, p, x, None, o, o, use_xyz=False) 
    center_x = x.reshape(-1,1,3).repeat(1,nsample,1)
    d_x = ((center_x - knn_x)**2).sum(-1).sum(-1) # N
    min_d = d_x.min(dim=-1)[0].repeat([d_x.size(0)])
    max_d = d_x.max(dim=-1)[0].repeat([d_x.size(0)])
    d_x = (d_x-min_d)/(max_d-min_d)

    # calculate the local difference of rgb
    knn_p = pointops.queryandgroup(nsample, p, p, p, None, o, o, use_xyz=False) 
    center_p = p.reshape(-1,1,3).repeat(1,nsample,1)
    d_p = ((center_p - knn_p)**2).sum(-1).sum(-1)
    min_d = d_p.min(dim=-1)[0].repeat([d_p.size(0)])
    max_d = d_p.max(dim=-1)[0].repeat([d_p.size(0)])
    d_p = (d_p-min_d)/(max_d-min_d)

    # local difference to denote point statistics
    d = d_x + d_p

    # mask process
    mask_p=[]
    mask_x=[]
    mask_o=[]
    count=0

    tar_p=[]
    tar_x=[]
    tar_o=[]
    tar_count=0

    for i in range(o.size(0)):
        p_i = p[o[i-1]:o[i],:] if i!=0  else p[:o[i],:]
        x_i = x[o[i-1]:o[i],:] if i!=0  else x[:o[i],:]
        d_i = d[o[i-1]:o[i]] if i!=0  else d[:o[i]]

        mask_rate = 0.2 + 0.7*torch.rand(1)
        idx = d_i.topk(k=int(mask_rate*p_i.size(0)), dim=0)[1]
        mask_p.append(p_i[idx])
        mask_x.append(x_i[idx])
        count += idx.size(0)
        mask_o.append(count)

        tar_rate= mask_rate + completion_gap
        tar_idx = d_i.topk(k=int(tar_rate*p_i.size(0)), dim=0)[1]
        tar_p.append(p_i[tar_idx])
        tar_x.append(x_i[tar_idx])
        tar_count += tar_idx.size(0)
        tar_o.append(tar_count)

    # masked scene
    mask_p = torch.cat(mask_p)
    mask_x = torch.cat(mask_x)
    mask_o = torch.IntTensor(mask_o).to(p.device)

    # corrseponding target scene
    tar_p = torch.cat(tar_p)
    tar_x = torch.cat(tar_x)
    tar_o = torch.IntTensor(tar_o).to(p.device)

    return [mask_p,mask_x,mask_o], [tar_p, tar_x, tar_o]





# completion decoder: use displacements feature to generate the displacements of scene corrdinates of the input scene
class Fold(nn.Module):
    def __init__(self, in_channel, step , hidden_dim = 512):
    '''
    in_channel: number of input channels
    step: number of steps for folding
    hidden_dim: number of hidden dimensions (default is 512)
    '''
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
        # x: Scene features [N,C]
        num_sample = self.step * self.step
        npos = x.size(0)
        features = x.view(npos, self.in_channel, 1).expand(npos, self.in_channel, num_sample) # [npos,C,num_sample]
        seed = self.folding_seed.view(1, 2, num_sample).expand(npos, 2, num_sample).to(x.device) # [npos,2, num_sample]

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2  # [npos,3,num_sample]


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        '''
        in_planes: number of input channels
        out_planes: number of output channels
        share_planes: number of shared planes (default is 8)
        nsample: number of samples (default is 16)
        '''
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

        self.completion_decoder_1 = Fold(planes[0]+3, step = 2, hidden_dim = 256)
        self.completion_decoder_2 = Fold(planes[1]+3, step = stride[1]//2, hidden_dim = 256)
        self.completion_decoder_3 = Fold(planes[2]+3, step = stride[2]//2, hidden_dim = 256)
        self.completion_decoder_4 = Fold(planes[3]+3, step = stride[3]//2, hidden_dim = 256)
        self.completion_decoder_5 = Fold(planes[4]+3, step = stride[4]//2, hidden_dim = 256)



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
        update_moving_average(self.target_ema_updater, self.target_encoder, self.PT_model)

    def forward(self, pxto):

        masked_pxto, tar_pxo = Informative_Perserved_Mask(pxto)


        pxo1,pxo2,pxo3,pxo4,pxo5 = self.PT_model(masked_pxto)
        p1,x1,o1 = pxo1
        p2,x2,o2 = pxo2
        p3,x3,o3 = pxo3
        p4,x4,o4 = pxo4
        p5,x5,o5 = pxo5

        ## ------------------
        x5_ = torch.cat([x5,p5], dim= -1)  
        c5 = (self.completion_decoder_5(x5_) + p5.unsqueeze(-1))

        x4_ = torch.cat([x4,p4], dim= -1)  
        c4 = (self.completion_decoder_4(x4_) + p4.unsqueeze(-1))

        x3_ = torch.cat([x3,p3], dim= -1)          
        c3 = (self.completion_decoder_3(x3_)+ p3.unsqueeze(-1))

        x2_ = torch.cat([x2,p2], dim= -1)  
        c2 = (self.completion_decoder_2(x2_) + p2.unsqueeze(-1))

        x1_ = torch.cat([x1,p1], dim= -1)  
        c1 = self.completion_decoder_1(x1_) + p1.unsqueeze(-1)




        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            tar_pxo1, tar_pxo2, tar_pxo3, tar_pxo4, tar_pxo5 = target_encoder(tar_pxo)
            tar_p1,tar_x1,tar_o1 = tar_pxo1
            tar_p2,tar_x2,tar_o2 = tar_pxo2
            tar_p3,tar_x3,tar_o3 = tar_pxo3
            tar_p4,tar_x4,tar_o4 = tar_pxo4
            tar_p5,tar_x5,tar_o5 = tar_pxo5

            ct_par_x1 = pointops.interpolation(tar_p1, p1, tar_x1, tar_o1, o1).detach_()
            ct_par_x2 = pointops.interpolation(tar_p2, p2, tar_x2, tar_o2, o2).detach_()
            ct_par_x3 = pointops.interpolation(tar_p3, p3, tar_x3, tar_o3, o3).detach_()
            ct_par_x4 = pointops.interpolation(tar_p4, p4, tar_x4, tar_o4, o4).detach_()
            ct_par_x5 = pointops.interpolation(tar_p5, p5, tar_x5, tar_o5, o5).detach_()


        return [c1,o1],[c2,o2],[c3,o3],[c4,o4],[c5,o5],tar_pxo,\
            [ct_par_x1,x1,o1],\
            [ct_par_x2,x2,o2],\
            [ct_par_x3,x3,o3],\
            [ct_par_x4,x4,o4],\
            [ct_par_x5,x5,o5]   




def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model
