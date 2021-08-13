import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..noisy_mixup import do_noisy_mixup

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width=1):
        super(PreActResNet, self).__init__()
        
        widths = [int(w * width) for w in [64, 128, 256, 512]]
        
        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(widths[3]*block.expansion, num_classes)
        self.blocks = [self.layer1, self.layer2, self.layer3, self.layer4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, targets=None, mixup_alpha=0.0, manifold_mixup=0, 
                add_noise_level=0.0, mult_noise_level=0.0):
           
        k = 0 if mixup_alpha > 0.0 else -1
        if mixup_alpha > 0.0 and manifold_mixup == True: k = np.random.choice(range(4), 1)[0]
        
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

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        if mixup_alpha > 0.0:
            return out, targets_a, targets_b, lam
        else:
            return out


def PreActResNet18(**kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], **kwargs)

def PreActResNetWide18(**kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], width=2, **kwargs)

preactresnet18 = PreActResNet18
preactwideresnet18 = PreActResNetWide18

def test():
    net = PreActResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

