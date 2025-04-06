import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, inchannel, outchannel, res=True, stride=1):
        super(Block, self).__init__()
        self.res = res     # 是否带残差连接
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512,'M'], res=True):
        super(ResNet, self).__init__()
        self.res = res       # 是否带残差连接
        self.cfg = cfg       # 配置列表
        self.inchannel = 3   # 初始输入通道数
        self.features = self.make_layer()
        # 构建卷积层之后的全连接层以及分类器：
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),           
            nn.Linear(4 * 512, 10),    # fc，最终Cifar10输出是10类
        )

    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v    # 输入通道数改为上一层的输出通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out 

class ResNet_MNIST(nn.Module):
    def __init__(self, cfg=[32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'], res=True):
        super(ResNet_MNIST, self).__init__()
        self.res = res       
        self.cfg = cfg       
        
        # 添加初始卷积层，使用更小的初始通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # 添加空间dropout
        )
        
        self.inchannel = 16
        self.features = self.make_layer()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 修改分类器，添加更多正则化
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v    # 输入通道数改为上一层的输出通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        out = self.features(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)