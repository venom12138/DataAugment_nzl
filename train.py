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
from networks.losses import MySupConLoss

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
parser.add_argument('--local_ckpt', type=str, default='')

parser.add_argument('--epochs', type=int, nargs='+', default=[100, 30])
parser.add_argument('--initial_learning_rate', type=float, nargs='+', default=[0.1])
parser.add_argument('--batch_size', type=int, nargs='+', default=[128])
parser.add_argument('--changing_lr', type=int, nargs='+', default=[80, 20])

parser.add_argument('--stage', type=int, default=None)  # None: baseline
parser.add_argument('--aux_config', type=str, default=None)

parser.add_argument('--feat_transform', type=str, nargs='+', default=[])
parser.add_argument('--criterion', type=str, default='cross_entropy')
parser.add_argument('--baseline_type', type=str, default=None,
                    choices=['rand', 'best'])  # run baseline exp with local train epoch=0
parser.add_argument('--local_ckpt_opt', type=str, default='last', choices=['last', 'best', 'path'])
args = parser.parse_args()

# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': 160,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [80, 120],
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
    print('-' * 10, f'{phase} phase', '-' * 10)
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

    feat_transform = []
    if phase == 'local_train' and args.stage != 1:
        if 'rand_crop' in args.feat_transform:
            if args.stage == 2:
                p = 4
                h = 32
            if args.stage == 3:
                p = 2
                h = 16
            feat_transform.extend([transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                                    (p, p, p, p), mode='reflect').squeeze()),
                                transforms.RandomCrop(h)])
        if 'flip' in args.feat_transform:
            feat_transform.append(transforms.RandomHorizontalFlip())
    feat_transform = transforms.Compose(feat_transform)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        myCIFAR10('data', feature_path='data/save_feature', train=True, download=True, transform=transform_train,
                    stage=args.stage, feat_transform=feat_transform),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        myCIFAR10('data', feature_path='data/save_feature', train=False, download=True, transform=transform_test,
                    stage=args.stage),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)

    # create model, giving aux_criterion to control feature_dim
    model = eval('networks.resnet.resnet' + str(args.layers) + '_cifar') \
        (dropout_rate=args.droprate, class_num=class_num,
            stage=args.stage, aux_config=args.aux_config, aux_criterion=args.criterion)

    cudnn.benchmark = True

    if phase == 'local_train' and args.criterion == 'contrast':
        criterion = MySupConLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])

    model = model.cuda()

    if args.baseline_type and phase == 'local_train':  # run baseline exp with local train epoch=0
        training_configurations[args.model]['epochs'] = 0
        if args.baseline_type == 'best':
            optim_checkpoint = torch.load(args.optim_ckpt)
            model.load_state_dict(optim_checkpoint['state_dict'], strict=False)
        save_checkpoint({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, True, checkpoint=exp.save_dir)

    if phase == 'finetune':
        optim_checkpoint = torch.load(args.optim_ckpt)
        if args.local_ckpt_opt == 'best':
            checkpoint = torch.load(exp.save_dir + '/model_best.pth.tar')
        elif args.local_ckpt_opt == 'last':
            checkpoint = torch.load(exp.save_dir + '/checkpoint.pth.tar')
        elif args.local_ckpt_opt == 'path':
            assert args.local_ckpt != ''
            checkpoint = torch.load(args.local_ckpt)
        else:
            raise NotImplementedError
        model.load_state_dict(optim_checkpoint['state_dict'], strict=False)  # aux_classifier出问题
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    for epoch in range(0, training_configurations[args.model]['epochs']):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train_metrics = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        eval_metrics, prec1 = validate(val_loader, model, criterion, epoch)

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
        if epoch in [training_configurations[args.model]['changing_lr']]:
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
    conf = {k: v if len(v) == 2 else v+v for k, v in vars(args).items() if k in keys2update}

    training_configurations['resnet'].update({k: v[0] for k, v in conf.items()})
    main(phase='local_train')

    # finetuning
    training_configurations['resnet'].update({k: v[1] for k, v in conf.items()})
    args.stage = None
    args.aux_config = None
    main(phase='finetune')

    exp.finish()