import argparse
import shutil
import os
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import logging
import sys
from models.resnet_models import resnet19
from models.VGG_models import VGGSNN
import data_loaders
from functions import *

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=10,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b',
                    '--batch-size',
                    default=40,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--load_path', 
                    type=str, 
                    default='', 
                    help='pretrained model path')
parser.add_argument('--seed',
                    default=1234,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--T',
                    default=4,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--local_rank', 
                    default=os.getenv('LOCAL_RANK', -1), 
                    type=int)
args = parser.parse_args()

def main():
    seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')
    logging.info("args = %s", args)


    # Initialize Process Group
    dist.init_process_group(backend='nccl')
    dist.barrier()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)


    model = resnet19(num_classes=10)
    model.T = args.T
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    criterion = nn.CrossEntropyLoss().to(device)

    # Data loading code
    train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)


    start_epoch = 0
    if args.load_path != None:
        logging.info(f"=> loading checkpoint {args.load_path}")
        ckpt = torch.load(args.load_path)
        model.load_state_dict(ckpt['net'])

    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()
        val_sampler.set_epoch(epoch)
        validate(val_loader, model, criterion, device, args)


def validate(val_loader, model, criterion, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        timestep_loss = {}
        timestep_add_loss = {}
        for idx in range(args.T):
            timestep_loss[idx] = []
            timestep_add_loss[idx] = []
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            for idx in range(args.T):
                timestep_loss[idx].append(criterion(output[:, idx, ...], target).detach().cpu())
                timestep_add_loss[idx].append(criterion(torch.mean(output[:, 0:(idx+1), ...], dim=1), target).detach().cpu())
            mean_out = torch.mean(output, dim=1)
            loss = criterion(mean_out, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        
        for t in range(args.T):
            tmp = np.array(output[:, t, ...].cpu())


        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))
    mean_out= np.array(mean_out.cpu())

    for idx in range(args.T):
        timestep_loss[idx] = np.array(timestep_loss[idx])
        timestep_loss[idx] = timestep_loss[idx].mean()
        timestep_add_loss[idx] = np.array(timestep_add_loss[idx])
        timestep_add_loss[idx] = timestep_add_loss[idx].mean()

def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")

def save_checkpoint(state, filename):
    filename = os.path.join(filename, 'checkpoint.pth.tar')
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
