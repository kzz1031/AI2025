import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

# CIFAR10数据集的类别标签
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

class CustomCIFAR10(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.transform = transform
        self.train = train
        
        if self.train:
            # 读取训练数据
            self.data = []
            self.targets = []
            for i in range(1, 6):
                file_path = os.path.join(root_dir, f'data_batch_{i}')
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    self.targets.extend(entry['labels'])
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        else:
            # 读取测试数据
            file_path = os.path.join(root_dir, 'test_batch')
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data = entry['data'].reshape(-1, 3, 32, 32)
                self.targets = entry['labels']
        
        self.data = self.data.transpose((0, 2, 3, 1))  # 转换为HWC格式

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

class CIFAR10Net(nn.Module):
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

def main():
    # 参数设置
    BATCH_SIZE = 1024
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 7
    VALIDATION_SPLIT = 0.1

    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载数据集
    full_train_dataset = CustomCIFAR10('./cifar-10-batches-py', train=True, transform=train_transform)
    #test_dataset = CustomCIFAR10('./cifar-10-batches-py', train=False, transform=test_transform)

    # 划分训练集和验证集
    val_size = int(len(full_train_dataset) * VALIDATION_SPLIT)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        accuracy = 100. * correct / len(val_loader.dataset)
        
        print(f'Validation set: Average loss: {val_loss:.4f}, '
              f'Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)')

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'cifar10_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping triggered after epoch {epoch}')
                break

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('cifar10_best.pth'))
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += criterion(output, target).item()
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader)
    # accuracy = 100. * correct / len(test_loader.dataset)
    # print(f'\nTest set: Average loss: {test_loss:.4f}, '
    #       f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

if __name__ == '__main__':
    main()
