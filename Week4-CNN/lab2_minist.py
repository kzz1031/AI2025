import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 添加这行,设置backend
import matplotlib.pyplot as plt

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

# parameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.01
PATIENCE = 5 
VALIDATION_SPLIT = 0.2  

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
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # 计算损失和准确率
        train_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 计算平均损失和准确率
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({train_accuracy:.2f}%)')
    
    return train_loss, train_accuracy

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
    wrong_samples = []  # 存储错误预测的样本
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            
            # 收集错误预测的样本
            wrong_mask = ~pred.eq(target.view_as(pred)).squeeze()
            wrong_samples.extend([
                (data[i].cpu().numpy().squeeze(), 
                 int(target[i].cpu().numpy().item()),
                 int(pred[i].cpu().numpy().item()))
                for i in range(len(data)) if wrong_mask[i]
            ])
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')
    
    return wrong_samples

def plot_wrong_predictions(wrong_samples, num_samples=15):
    """显示错误预测的样本"""
    num_samples = min(num_samples, len(wrong_samples))
    fig, axes = plt.subplots(3, 5, figsize=(15, 6))
    
    for i, (img, true_label, pred_label) in enumerate(wrong_samples[:num_samples]):
        ax = axes[i//5, i%5]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True:{true_label}\nPred:{pred_label}')
        ax.axis('off')
    
    plt.tight_layout()
    # 保存图片而不是显示
    plt.savefig('wrong_predictions.png')
    plt.close()
    print("saved as: wrong_predictions.png")

# 训练模型
if __name__ == '__main__':
    print("Train or Test?(t/T for test, other for train)")
    choice = input()
    if choice == 't' or choice == 'T':
        print("Loading best model and testing...")
        model.load_state_dict(torch.load('best_model_minist.pth'))
        wrong_samples = test(model, device, test_loader)
        print('\nDisplaying some wrong predictions:')
        plot_wrong_predictions(wrong_samples)
    else:
        print("With or without dropout?(y/Y for with, other for without)")
        choice = input()
        if choice == 'y' or choice == 'Y':
            model.dropout1 = nn.Dropout(0.25)
            model.dropout2 = nn.Dropout(0.5)
        else:
            model.dropout1 = nn.Dropout(0)
            model.dropout2 = nn.Dropout(0)
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 记录训练过程中的各种指标
        train_losses = []
        val_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(1, EPOCHS + 1):
            # 使用训练函数并获取返回的指标
            train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # 验证模型
            val_loss = validate(model, device, val_loader)
            val_losses.append(val_loss)
            
            # 测试模型
            model.eval()
            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
            
            test_loss /= len(test_loader.dataset)
            test_accuracy = 100. * test_correct / len(test_loader.dataset)
            test_accuracies.append(test_accuracy)
            print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)\n')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model_minist.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f'Early stopping triggered after epoch {epoch}')
                break
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curves.png')
        plt.close()
        print("saved as: loss_curves.png")
        
        # 绘制准确率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Test Accuracy Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('accuracy_curves.png')
        plt.close()
        print("saved as: accuracy_curves.png")
        
        print('Loading best model and testing...')
        model.load_state_dict(torch.load('best_model_minist.pth'))
        wrong_samples = test(model, device, test_loader)
        print('\nDisplaying some wrong predictions:')
        plot_wrong_predictions(wrong_samples)
        print('===Finished Training===')
        test(model, device, test_loader)

# result
# 测试集准确率：99.08%（有dropout）
# 测试集准确率：98.86%（无dropout）