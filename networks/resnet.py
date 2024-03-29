'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''
import copy

from .losses import SupConLoss
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out

class AuxClassifier(nn.Module):
    def __init__(self, inplanes, net_config='1c2f', loss_mode='contrast', class_num=10, widen=1, feature_dim=128):
        super(AuxClassifier, self).__init__()

        assert inplanes in [16, 32, 64]
        assert net_config in ['0c1f', '0c2f', '1c1f', '1c2f', '1c3f', '2c2f']
        assert loss_mode in ['contrast', 'cross_entropy']

        self.loss_mode = loss_mode
        self.feature_dim = feature_dim

        if loss_mode == 'contrast':
            self.fc_out_channels = feature_dim
        elif loss_mode == 'cross_entropy':
            self.fc_out_channels = class_num
        else:
            raise NotImplementedError

        if net_config == '0c1f':  # Greedy Supervised Learning (Greedy SL)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(inplanes, self.fc_out_channels),
            )

        if net_config == '0c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(16, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '1c1f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), self.fc_out_channels),
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), self.fc_out_channels),
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), self.fc_out_channels),
                )

        if net_config == '1c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '1c3f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

        if net_config == '2c2f':
            if inplanes == 16:
                self.head = nn.Sequential(
                    nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(32 * widen), int(32 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(32 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(32 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 32:
                self.head = nn.Sequential(
                    nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(64 * widen), int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )
            elif inplanes == 64:
                self.head = nn.Sequential(
                    nn.Conv2d(64, int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.Conv2d(int(64 * widen), int(64 * widen), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(int(64 * widen)),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(64 * widen), int(feature_dim * widen)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(feature_dim * widen), self.fc_out_channels)
                )

    def forward(self, x):

        features = self.head(x)

        # if self.loss_mode == 'contrast':
        #     assert features.size(1) == self.feature_dim
        #     features = F.normalize(features, dim=1)
        #     features = features.unsqueeze(1)
        #     loss = self.criterion(features, target, temperature=0.07)
        # elif self.loss_mode == 'cross_entropy':
        #     loss = self.criterion(features, target)
        # else:
        #     raise NotImplementedError

        return features

class AuxClassifierNew(nn.Module):
    def __init__(self, inplanes, class_num=10, widen=1, feature_dim=128):
        super(AuxClassifierNew, self).__init__()

        assert inplanes in [16, 32]

        self.feature_dim = feature_dim

        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = class_num

        if inplanes == 16:
            self.head = nn.Sequential(
                nn.Conv2d(16, int(32 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(int(32 * widen)),
                nn.ReLU(),
                nn.Conv2d(int(32 * widen), int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(int(64 * widen)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(int(64 * widen), self.fc_out_channels),
            )
        elif inplanes == 32:
            self.head = nn.Sequential(
                nn.Conv2d(32, int(64 * widen), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(int(64 * widen)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(int(64 * widen), self.fc_out_channels),
            )

    def forward(self, x):
        features = self.head(x)
        return features


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, dropout_rate=0, class_num=10, stage=None
                 , aux_config=None, aux_criterion='cross_entropy'):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.dropout_rate = dropout_rate
        self.feature_num = 64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.stage = stage
        if stage == 1 or stage == 2:
            assert aux_config is not None
            if aux_config == 'new':
                self.aux_classifier = AuxClassifierNew(inplanes=16 if stage == 1 else 32, class_num=class_num)
            else:
                self.aux_classifier = AuxClassifier(inplanes=16 if stage == 1 else 32, net_config=aux_config,
                                                    loss_mode=aux_criterion, class_num=class_num,
                                                    )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.stage == None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            output = self.fc(x)
        elif self.stage == 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            output = self.aux_classifier(x)
        elif self.stage == 2:
            x = self.layer2(x)
            output = self.aux_classifier(x)
        elif self.stage == 3:
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            output = self.fc(x)

        return output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = nn.Module.state_dict(self)
        keys = list(state_dict.keys())
        if self.stage is not None:
            for k in keys:
                if self.stage == 1:
                    if k.startswith('conv1') or k.startswith('bn1') or k.startswith('layer1'):
                        continue
                elif self.stage == 2:
                    if k.startswith('layer2'):
                        continue
                elif self.stage == 3:
                    if k.startswith('layer3') or k.startswith('fc'):
                        continue
                state_dict.pop(k)
        return state_dict

    def gen_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        feat1 = self.layer1(x)  # feature map1
        feat2 = self.layer2(feat1)  # feature map2

        return x, feat1, feat2


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet20_mnist(**kwargs):
    model = ResNet_MNIST(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet20_cifar()
    y = net(torch.randn(1, 3, 64, 64))
    print(net)
    print(y.size())