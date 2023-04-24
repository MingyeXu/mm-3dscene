# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from votenet import cal_loss

import torch.nn.functional as F

from functools import wraps
import copy


class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        bsz = x.shape[0]
        x = x.squeeze()
        loss = self.criterion(x, label)
        return loss
        
def global_contrast_loss_fuc(feat1,feat2):
    loss = compute_loss(feat1, feat2, mask=None)

    return loss/feat1.size(0)
    
LARGE_NUM = 1e9
def compute_loss(q, k, mask=None):
    npos = q.shape[0]
    q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
    logits = torch.mm(q, k.transpose(1, 0)) # npos by npos
    # print("logtis", logits)
    labels = torch.arange(npos).cuda().long()
    out = torch.div(logits, 0.07)
    out = out.squeeze().contiguous()
    if mask != None:
      out = out - LARGE_NUM * mask.float()
    criterion = NCESoftmaxLoss().cuda()
    loss = criterion(out, labels)
    return loss



def contrast_loss_fuc(feat1,feat2):
    loss = 0
    # SELECTED_NUM = 4096
    for i in range(feat1.size(0)):
        f1 = feat1[i,:,:]
        f2 = feat2[i,:,:]
        # print(f1.size())
        # if f1.shape[0]> SELECTED_NUM:
        #     sampled_inds = np.random.choice(f1.shape[0], SELECTED_NUM, replace=False)
        #     f1 = f1[sampled_inds]
        #     f2 = f2[sampled_inds]
        loss+=compute_loss(f1,f2)
    return loss/feat1.size(0)


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='250,350,450', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.5,0.5,0.5', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
    TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
else:
    Detector = MODEL.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

if torch.cuda.device_count() > 1:
  log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)
net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch(epoch):
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    net._get_target_encoder()

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        mask_rate = 0.2 + 0.6*torch.rand(1)
        pred_xyz,gt_xyz,gt_feat, ct_feat,feat = net(inputs,mask_rate)
        loss = cal_loss( pred_xyz,gt_xyz) + contrast_loss_fuc(feat,ct_feat)
        # Compute loss and gradients, update parameters.
        # for key in batch_data_label:
        #     assert(key not in end_points)
        #     end_points[key] = batch_data_label[key]
        # loss, end_points = criterion(end_points, DATASET_CONFIG)


        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        net.update_moving_average()

        if (batch_idx + 1) %10 == 0 :
            log_string('Epoch: [{}][{}/{}] '
                        'Loss {loss_meter:.4f} '.format(epoch+1,  batch_idx + 1, len(TRAIN_DATALOADER),
                                                          loss_meter=loss.item()))

        # Accumulate statistics and print out
        # for key in end_points:
        #     if 'loss' in key or 'acc' in key or 'ratio' in key:
        #         if key not in stat_dict: stat_dict[key] = 0
        #         stat_dict[key] += end_points[key].item()

        # stat_dict[loss] += loss.item()

        # batch_interval = 10
        # if (batch_idx+1) % batch_interval == 0:
        #     log_string(' ---- batch: %03d ----' % (batch_idx+1))
        #     TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
        #         (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
        #     for key in sorted(stat_dict.keys()):
        #         log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
        #         stat_dict[key] = 0


def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch(epoch)
        # if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
        #     loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
        if epoch%10==0:
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint_'+str(epoch)+'.tar'))

if __name__=='__main__':
    train(start_epoch)
