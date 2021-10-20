import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import transforms
import torchvision.datasets as datasets
import os

import networks.resnet
import networks.wideresnet
import networks.se_resnet
import networks.se_wideresnet
import networks.densenet_bc
import networks.shake_pyramidnet
import networks.resnext
import networks.shake_shake

import numpy as np
resume_path = '/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/ISDA test/cifar10_resnet-32_feature_extract/no_0_lambda_0_0.5_standard-Aug_/checkpoint/model_best.pth.tar'
save_path = '/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path+'/feature1'):
    os.makedirs(save_path+'/feature1')
if not os.path.exists(save_path+'/feature2'):
    os.makedirs(save_path+'/feature2')
if not os.path.exists(save_path+'/feature3'):
    os.makedirs(save_path+'/feature3')
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
data_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
kwargs = {'num_workers': 1, 'pin_memory': True}
data_loader = torch.utils.data.DataLoader(
        datasets.__dict__['cifar10'.upper()]('../data', train=True, download=True,
                        transform=data_transform),
        batch_size=1, shuffle=False, **kwargs)
    
model = eval('networks.resnet.resnet' + str(32) + '_cifar')(dropout_rate=0.0)
model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load(resume_path)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
feature1s = torch.tensor([]).cuda()
feature2s = torch.tensor([]).cuda()
feature3s = torch.tensor([]).cuda()
for i, (input, target) in enumerate(data_loader):
    print(i)
    target = target.cuda()
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    # compute output
    with torch.no_grad():
        feature1, feature2, feature3 = model.module.gen_feature(input_var)
    feature1s = torch.cat((feature1s, feature1),dim = 0)
    feature2s = torch.cat((feature2s, feature2),dim = 0)
    feature3s = torch.cat((feature3s, feature3),dim = 0)
        # 50000,16,32,32
        # 50000,32,16,16
        # 50000,64,8 ,8
    # with open('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature1/'+str(i)+'.npy', 'w') as f1: 
    #     np.save('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature1/'+str(i)+'.npy', feature1.detach().cpu().numpy().tolist())
    # with open('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature2/'+str(i)+'.npy', 'w') as f1: 
    #     np.save('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature2/'+str(i)+'.npy', feature2.detach().cpu().numpy().tolist())
    # with open('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature3/'+str(i)+'.npy', 'w') as f1: 
    #     np.save('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature3/'+str(i)+'.npy', feature3.detach().cpu().numpy().tolist())
# print(feature1s.shape)
with open('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature1'+'.npy', 'w') as f1: 
    np.save('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature1'+'.npy', feature1s.detach().cpu().numpy().tolist())
with open('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature2'+'.npy', 'w') as f1: 
    np.save('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature2'+'.npy', feature2s.detach().cpu().numpy().tolist())
with open('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature3'+'.npy', 'w') as f1: 
    np.save('/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature3'+'.npy', feature3s.detach().cpu().numpy().tolist())
