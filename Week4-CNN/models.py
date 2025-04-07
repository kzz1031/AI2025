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
    支持手动设置卷积核大小
    """
    def __init__(self, kernel_size=3):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        
        # 动态计算全连接层的输入大小
        self._calculate_fc_input_size()
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _calculate_fc_input_size(self):
        # 创建一个虚拟张量，计算经过卷积层后的大小
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)  # MNIST 输入大小 (1, 28, 28)
            x = self.conv1(x)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            self.fc_input_size = x.numel()  # 计算展平后的大小

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
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 10)

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
        self.block3 = Conv_Block(inchannel=128, outchannel=256, res=res)
        self.block4 = Conv_Block(inchannel=256, outchannel=512, res=res)
        
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
        out = self.maxpool(out)
        out = self.block3(out)
        out = self.maxpool(out)
        out = self.block4(out)
        out = self.maxpool(out)
        out = self.classifier(out)
        return out
    
class ResNet_CIFAR10_PLUS(nn.Module):
    """
    CIFAR-10数据集的残差网络模型
    包含7个残差块和全连接分类器
    """
    def __init__(self, res=False):
        super(ResNet_CIFAR10_PLUS, self).__init__()
 
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

class VGG19_CIFAR10(nn.Module):
    """
    VGG19网络模型，针对CIFAR-10数据集优化
    包含5个卷积块和3个全连接层，使用批归一化和dropout
    """
    def __init__(self, dropout_rate=0.5, weight_decay=0.0001):
        super(VGG19_CIFAR10, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 5
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x)
        return x

def make_vgg_layers(spec):
    """构建VGG特征提取层"""
    layers = []
    n_chans_in = 3
    for v in spec:
        if isinstance(v, int):
            # 批归一化所以不需要偏置
            layers += [
                nn.Conv2d(n_chans_in, v, 3, padding=1, bias=False),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True)
            ]
            n_chans_in = v
        elif v == "M":
            layers += [nn.MaxPool2d(2)]
    return nn.Sequential(*layers)

class VGG16_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR10, self).__init__()
        
        # 特征提取层配置
        feature_layers = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512, "M",
            512, 512, 512, "M",
        ]
        
        # 构建特征提取层
        self.features = make_vgg_layers(feature_layers)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x