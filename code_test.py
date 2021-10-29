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

import matplotlib.pyplot as plt
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

parser.add_argument('--layers', default=32, type=int,
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

parser.add_argument('--finetune', type=int, default=0)
parser.add_argument('--optim_ckpt', type=str, default='') # ?
parser.add_argument('--local_module_ckpt', type=str, default='')

parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--initial_learning_rate', type=float, default=0.1)
parser.add_argument('--changing_lr', type=int, nargs="+", default=[80, 120])

parser.add_argument('--stage', type=int, default=None)  # None: baseline
parser.add_argument('--aux_config', type=str, default=None)
parser.add_argument('--mix_epoch', type=int, default=10000, help = 'train stage within mixepoch then train all stages')
parser.add_argument('--mix_mode', type=str, default='', help = 'stochastic or optim')
args = parser.parse_args()

training_configurations = {
    'resnet': {
        'batch_size': 128,
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
}
training_configurations['resnet'].update(vars(args))
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
model = eval('networks.resnet.resnet' + str(args.layers) + '_cifar')\
        (dropout_rate=args.droprate, class_num=10,
         stage=args.stage, aux_config=args.aux_config)
kwargs = {'num_workers': 1, 'pin_memory': True}
transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ])
train_loader = torch.utils.data.DataLoader(
        myCIFAR10('data', feature_path = 'data/save_feature', train=True, download=True, transform=transform_train,
                  stage=args.stage),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
params = {}
params.update({'shit':torch.tensor([1,2,3.0])})
optimizer = torch.optim.SGD(model.parameters(),
                        lr=training_configurations[args.model]['initial_learning_rate'],
                        momentum=training_configurations[args.model]['momentum'],
                        nesterov=training_configurations[args.model]['nesterov'],
                        weight_decay=training_configurations[args.model]['weight_decay'])
def main():
    lr = []
    t = []
    for epoch in range(training_configurations['resnet']['epochs']):
        lr.append(adjust_learning_rate(optimizer, epoch))
        t.append(epoch)
        # print(lr)
    plt.figure()
    plt.plot(t,lr)
    plt.show()
    plt.savefig('./lr1.png')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']

    else:
        if args.mix_epoch < training_configurations[args.model]['epochs']:
            if epoch < args.mix_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                        * (1 + math.cos(math.pi * epoch / args.mix_epoch))
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                        * (1 + math.cos(math.pi * (epoch-args.mix_epoch) / (training_configurations[args.model]['epochs']-args.mix_epoch)))
        else:
            for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                        * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))
    return(param_group['lr'])
if __name__ == '__main__':
    main()