import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os

class CustomMNIST(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        if self.train:
            image_file = os.path.join(root_dir, 'train-images.idx3-ubyte')
            label_file = os.path.join(root_dir, 'train-labels.idx1-ubyte')
        else:
            image_file = os.path.join(root_dir, 't10k-images.idx3-ubyte')
            label_file = os.path.join(root_dir, 't10k-labels.idx1-ubyte')
            
        self.images = self._read_images(image_file)
        self.labels = self._read_labels(label_file)

    def _read_images(self, file_path):
        with open(file_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            image_data = image_data.reshape(num_images, rows, cols)
            return image_data

    def _read_labels(self, file_path):
        with open(file_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            
            label_data = np.frombuffer(f.read(), dtype=np.uint8)
            return label_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image.astype(np.float32) / 255.0)
        
        return image, label
    
torch.manual_seed(42)

# 定义超参数
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.01
PATIENCE = 5  # 早停耐心值
VALIDATION_SPLIT = 0.1  # 验证集比例

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载和划分数据集
full_train_dataset = CustomMNIST('./minist_train_data', train=True, transform=transform)
test_dataset = CustomMNIST('./minist_train_data', train=False, transform=transform)

# 计算验证集大小
val_size = int(len(full_train_dataset) * VALIDATION_SPLIT)
train_size = len(full_train_dataset) - val_size

# 随机划分训练集和验证集
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
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

# 创建模型实例
if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.5)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 验证函数
def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation set: Average loss: {val_loss:.4f}, '
          f'Accuracy: {correct}/{len(val_loader.dataset)} '
          f'({val_accuracy:.2f}%)')
    return val_loss

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

# 训练模型
if __name__ == '__main__':
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, val_loader)
        test(model, device, test_loader)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f'Early stopping triggered after epoch {epoch}')
            break