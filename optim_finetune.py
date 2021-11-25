import argparse
import os
import shutil
import time
import errno
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import wandb

import transforms
import torchvision.datasets as datasets
import networks.resnet
from utils import ExpHandler
from collections import OrderedDict
from mydataset import myCIFAR10

parser = argparse.ArgumentParser(description='Implicit Semantic Data Augmentation (ISDA)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')

parser.add_argument('--model', default='resnet', type=str,
                    help='deep networks to be trained')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--layers', default=56, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=True)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.set_defaults(resume=False)

# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=False)
parser.add_argument('--en_wandb', action='store_true')

parser.add_argument('--optim_ckpt', type=str, default='')

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--initial_learning_rate', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=1024)

parser.add_argument('--stage', type=int, default=None)  # None: baseline
parser.add_argument('--aux_config', type=str, default=None)

args = parser.parse_args()

# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': 160,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
}
exp = ExpHandler(args.en_wandb)
exp.save_config(args)

if args.en_wandb:
    wandb.define_metric('local_train/eval_top1', summary='max')
    wandb.define_metric('finetune/eval_top1', summary='max')


def main(phase):
    global best_prec1

    best_prec1 = 0

    global class_num

    class_num = args.dataset == 'cifar10' and 10 or 100

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        print('Standrad Augmentation!')
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        myCIFAR10('data', feature_path = 'data/save_feature', train=True, download=True, transform=transform_train,
                    stage=args.stage),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        myCIFAR10('data', feature_path = 'data/save_feature', train=False, download=True, transform=transform_test,
                stage=args.stage),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)

    # create model
    model = eval('networks.resnet.resnet' + str(args.layers) + '_cifar')\
        (dropout_rate=args.droprate, class_num=class_num,
            stage=args.stage, aux_config=args.aux_config)


    cudnn.benchmark = True

    ce_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])

    model = model.cuda()
    optim_checkpoint = torch.load(args.optim_ckpt)
    model.load_state_dict(optim_checkpoint['state_dict'])  # aux_classifier出问题

    for epoch in range(0, training_configurations[args.model]['epochs']):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train_metrics = train(train_loader, model, ce_criterion, optimizer, epoch)

        # evaluate on validation set
        eval_metrics, prec1 = validate(val_loader, model, ce_criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=exp.save_dir)
        print('Best accuracy: ', best_prec1)

        exp.write(phase, eval_metrics, train_metrics, epoch=epoch,
                  lr=optimizer.param_groups[0]['lr'])

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        target = target.cuda()
        x = x.cuda()

        output = model(x)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            # print(discriminate_weights)
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))

            exp.log(string)

    return OrderedDict(loss=losses.ave, top1=top1.ave)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    exp.log(string)
    return OrderedDict(loss=losses.ave, top1=top1.ave), top1.ave


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']

    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    keys2update = vars(args).keys() & training_configurations['resnet'].keys()

    # finetuning
    training_configurations['resnet'].update({k: vars(args)[k] for k in keys2update})
    main(phase='finetune')

    exp.finish()