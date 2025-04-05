import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Net(nn.Module):
    """
    CIFAR-10数据集的CNN模型
    包含4个卷积层和2个全连接层，使用批归一化和dropout
    """
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class MNISTNet(nn.Module):
    """
    MNIST数据集的CNN模型
    包含2个卷积层和2个全连接层，使用dropout进行正则化
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 

class CIFAR10NET_VGG_MINI(nn.Module):
    def __init__(self):
        super(CIFAR10NET_VGG_MINI, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512 * 2 * 2, 10)
        #self.fc2 = nn.Linear(1024, 10)

class Conv_Block(nn.Module):
    """
    残差网络的基本构建块，包含两个卷积层和可选的残差连接
    """
    def __init__(self, inchannel, outchannel, res=True):
        super(Conv_Block, self).__init__()
        self.res = res  # 是否带残差连接
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet_CIFAR10(nn.Module):
    """
    CIFAR-10数据集的残差网络模型
    包含4个残差块和全连接分类器
    """
    def __init__(self, res=False):
        super(ResNet_CIFAR10, self).__init__()
 
        self.block1 = Conv_Block(inchannel=3, outchannel=64, res=res)
        self.block2 = Conv_Block(inchannel=64, outchannel=128, res=res)
        self.block3 = Conv_Block(inchannel=128, outchannel=128, res=res)
        self.block4 = Conv_Block(inchannel=128, outchannel=256, res=res)
        self.block5 = Conv_Block(inchannel=256, outchannel=256, res=res)
        self.block6 = Conv_Block(inchannel=256, outchannel=512, res=res)
        self.block7 = Conv_Block(inchannel=512, outchannel=512, res=res)
        
        # 构建卷积层之后的全连接层以及分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512 * 2 * 2, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 10)
        )
 
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
 
    def forward(self, x):
        out = self.block1(x)
        out = self.maxpool(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.maxpool(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.maxpool(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.maxpool(out)
        out = self.classifier(out)
        return out