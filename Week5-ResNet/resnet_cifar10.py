import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from models import ResNet  # 导入模型定义

# 使用torchvision可以很方便地下载Cifar10数据集
norm_mean = [0.485, 0.456, 0.406]  # 均值
norm_std = [0.229, 0.224, 0.225]  # 方差
transform_train = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(norm_mean, norm_std),
    transforms.RandomHorizontalFlip(),
    transforms.RandomErasing(p=0.5), 
    transforms.RandomCrop(32, padding=4)  
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 超参数：
batch_size = 256
num_epochs = 200   # 训练轮数
LR = 0.01          # 初始学习率

# 选择数据集:
trainset = datasets.CIFAR10(root='Datasets', train=True, download=False, transform=transform_train)
testset = datasets.CIFAR10(root='Datasets', train=False, download=False, transform=transform_test)
# 加载数据:
train_data = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
valid_data = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_data_size = len(trainset)
valid_data_size = len(testset)

print('train_size: {:4d}  valid_size:{:4d}'.format(train_data_size, valid_data_size))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = ResNet(res=True)  # 使用带残差连接的ResNet

# 定义损失函数和优化器
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-3)

# 学习率调整策略 MultiStep：
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                         milestones=[int(num_epochs * 0.56), int(num_epochs * 0.78)],
                                         gamma=0.1, last_epoch=-1)

def evaluate(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train_and_valid(model, loss_function, optimizer, epochs=10):
    model.to(device)
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        # 在每个epoch结束后评估测试集
        valid_loss, valid_acc = evaluate(model, valid_data, loss_function, device)
        
        # 更新学习率
        scheduler.step()
        print('\t当前学习率:', scheduler.get_last_lr()[0])

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        history.append([avg_train_loss, valid_loss, avg_train_acc, valid_acc/100.0])

        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            # 保存最佳模型
            torch.save(model.state_dict(), 'resnet_best_model_cifar10.pth')

        epoch_end = time.time()

        print(
            "\t训练集: Loss: {:.4f}, Accuracy: {:.2f}%, "
            "\n\t测试集: Loss: {:.4f}, Accuracy: {:.2f}%, 用时: {:.3f}s".format(
                avg_train_loss, avg_train_acc * 100, valid_loss, valid_acc,
                epoch_end - epoch_start
            ))
        print("\t当前最佳测试集准确率: {:.2f}% (Epoch {:03d})".format(best_acc, best_epoch))

    return model, history

def test_model(model_path='resnet_best_model.pth'):
    """
    直接加载模型进行测试
    Args:
        model_path: 模型权重文件路径
    """
    # 创建模型实例
    model = ResNet(res=True).to(device)
    
    try:
        print(f"正在加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path))
        print("模型加载成功")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}")
        return
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return
    
    # 评估模型
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    print("开始测试...")
    with torch.no_grad():
        for data, target in valid_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 计算每个类别的准确率
            correct_tensor = pred.eq(target.view_as(pred))
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += correct_tensor[i].item()
                class_total[label] += 1

    test_loss /= len(valid_data.dataset)
    accuracy = 100. * correct / len(valid_data.dataset)
    
    # 打印总体结果
    print('\n测试结果:')
    print(f'平均损失: {test_loss:.4f}')
    print(f'总体准确率: {correct}/{len(valid_data.dataset)} ({accuracy:.2f}%)\n')
    
    # 打印每个类别的准确率
    print('各类别准确率:')
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'{cifar10_classes[i]}: {class_acc:.2f}%')

if __name__ == '__main__':
    print("训练还是测试? (t/T 表示测试，其他表示训练)")
    choice = input()
    
    if choice.lower() == 't':
        print("测试模型...")
        model_path = 'models/resnet_best_model_cifar10.pth'    
        test_model(model_path)
    else:
        # 开始训练
        all_start = time.time()
        trained_model, history = train_and_valid(model, loss_func, optimizer, num_epochs)

        # 绘制损失曲线
        history = np.array(history)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history[:, 0:2])
        plt.legend(['Train Loss', 'Valid Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history[:, 2:4])
        plt.legend(['Train Accuracy', 'Valid Accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()

        all_end = time.time()
        all_time = round(all_end - all_start)
        print('总训练时间: {:d} 分 {:d} 秒'.format(all_time // 60, all_time % 60))

        # 加载最佳模型进行最终测试
        print('加载最佳模型进行最终测试...')
        test_model('resnet_best_model.pth')

