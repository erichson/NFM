import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np
import sys

from ..noisy_mixup import do_noisy_mixup


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.blocks = [self.layer1, self.layer2, self.layer3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, targets=None, mixup_alpha=0.0, manifold_mixup=0, 
                add_noise_level=0.0, mult_noise_level=0.0):
    
        k = 0 if mixup_alpha > 0.0 else -1
        if mixup_alpha > 0.0 and manifold_mixup == True: k = np.random.choice(range(3), 1)[0]
        
        if k == 0: # Do input mixup if k is 0 
          x, targets_a, targets_b, lam = do_noisy_mixup(x, targets, alpha=mixup_alpha, 
                                              add_noise_level=add_noise_level, 
                                              mult_noise_level=mult_noise_level)
    
    
        out = self.conv1(x)

        for i, ResidualBlock in enumerate(self.blocks):
            out = ResidualBlock(out)
            if k == (i+1): # Do manifold mixup if k is greater 0
                out, targets_a, targets_b, lam = do_noisy_mixup(out, targets, alpha=mixup_alpha, 
                                           add_noise_level=add_noise_level, 
                                           mult_noise_level=mult_noise_level)        
        

        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if mixup_alpha > 0.0:
            return out, targets_a, targets_b, lam
        else:
            return out

def WideResNet28(**kwargs):
    return Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10)

wideresnet28 = WideResNet28

def test():
    net = WideResNet()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

