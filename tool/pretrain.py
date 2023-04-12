import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
from torch.autograd import gradcheck
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from util import config
from util.dataset_pretrain import DatasetPretrain
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port,NCESoftmaxLoss
from util.data_util import collate_fn, collate_fn_pretrain
from util import transform as t
import scipy.io as sio

from model.pointtransformer.pointtransformer_pretrain import cal_loss

from torch import nn
import torch.nn.functional as F





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
    labels = torch.arange(npos).cuda().long()
    out = torch.div(logits, args.get('nceT', 0.07))
    out = out.squeeze().contiguous()
    if mask != None:
      out = out - LARGE_NUM * mask.float()
    criterion = NCESoftmaxLoss().cuda()
    loss = criterion(out, labels)
    return loss



# from lib.chamfer_dist import ChamferDistanceL1
CUDA_LAUNCH_BLOCKING=1

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def contrast_loss_fuc(feat1,feat2,offset):
    loss = 0
    SELECTED_NUM = 4096
    for i in range(offset.size(0)):
        f1 = feat1[:offset[i],:] if i==0 else feat1[offset[i-1]:offset[i],:]
        f2 = feat2[:offset[i],:] if i==0 else feat2[offset[i-1]:offset[i],:]
        if f1.shape[0]> SELECTED_NUM:
            sampled_inds = np.random.choice(f1.shape[0], SELECTED_NUM, replace=False)
            f1 = f1[sampled_inds]
            f2 = f2[sampled_inds]
        loss+=compute_loss(f1,f2)
    return loss/offset.size(0)


def global_contrast_loss_fuc(feat1,feat2):
    loss = compute_loss(feat1, feat2, mask=None)

    return loss/feat1.size(0)


LARGE_NUM = 1e9
def compute_loss(q, k, mask=None):
    npos = q.shape[0]
    q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
    logits = torch.mm(q, k.transpose(1, 0)) # npos by npos
    labels = torch.arange(npos).cuda().long()
    out = torch.div(logits, args.get('nceT', 0.07))
    out = out.squeeze().contiguous()
    if mask != None:
      out = out - LARGE_NUM * mask.float()
    criterion = NCESoftmaxLoss().cuda()
    loss = criterion(out, labels)
    return loss

def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    from model.pointtransformer.pointtransformer_pretrain  import pointtransformer_seg_repro as Model
        

    model = Model(c=args.fea_dim)
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()


   
  
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.base_lr, weight_decay=0.0005)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )
    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            #best_iou = 40.0
            # best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    train_data = DatasetPretrain(split='train',pretrain_dataset=args.data_name, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn_pretrain)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, filename)
            # , 'best_iou': best_iou, 'is_best': is_best
            # if is_best:
            #     logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
            #     shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    loss_mse =nn.MSELoss()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    model.module._get_target_encoder()

    for i, (scene_coor,scene_rgb,scene_offsets) in enumerate(train_loader):  # (n, 3), (n, c), (n) 
        data_time.update(time.time() - end)
        scene_coor, scene_rgb= scene_coor.cuda(non_blocking=True), scene_rgb.cuda(non_blocking=True)  
        scene_offsets = scene_offsets.cuda(non_blocking=True)
        c1,c2,c3,c4,c5, gt_pxo, ct_1,ct_2,ct_3,ct_4,ct_5 = model( [scene_coor, scene_rgb, scene_offsets])
        contrast_loss =  contrast_loss_fuc(ct_1[0],ct_1[1],ct_1[2])
        contrast_loss += contrast_loss_fuc(ct_2[0],ct_2[1],ct_2[2])
        contrast_loss += contrast_loss_fuc(ct_3[0],ct_3[1],ct_3[2])
        contrast_loss += contrast_loss_fuc(ct_4[0],ct_4[1],ct_4[2])
        contrast_loss += contrast_loss_fuc(ct_5[0],ct_5[1],ct_5[2])
        org_coor = gt_pxo[0]
        org_offset = gt_pxo[2]

        loss = cal_loss(c1,c2,c3,c4,c5,org_coor,org_offset)  + contrast_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.module.update_moving_average()
        n = scene_coor.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = scene_coor.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        accuracy = 0
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc =0
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
