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
from models import CIFAR10Net, ResNet_CIFAR10, VGG19_CIFAR10, VGG16_CIFAR10  # 导入模型定义

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

def main():
    # 基础参数设置
    VALIDATION_SPLIT = 0.1
    PATIENCE = 20
    EPOCHS = 200  # 增加训练轮数
    T_MAX = 100  # 余弦退火周期
    print("Train or Test?(t/T for test, other for train)")
    choice = input()
    
    # 定义测试数据预处理
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 选择模型
    print("Choose model type:")
    print("1: Standard CNN")
    print("2: ResNet")
    print("3: VGG19")
    print("4: VGG16")
    model_type_choice = input()
        
    if choice != 't' and choice != 'T':
        # 根据不同模型类型设置不同的训练参数
        if model_type_choice == '3':  # VGG19
            BATCH_SIZE = 128
            LEARNING_RATE = 0.1
            WEIGHT_DECAY = 5e-4
            MOMENTUM = 0.9
            
            # 学习率阶段设置
            WARMUP_EPOCHS = 5
            INITIAL_STAGE = 60
            FINAL_STAGE = 140
            
            print("Using VGG19 specific parameters:")
            print(f"Batch Size: {BATCH_SIZE}")
            print(f"Epochs: {EPOCHS}")
            print(f"Initial Learning Rate: {LEARNING_RATE}")
            print(f"Weight Decay: {WEIGHT_DECAY}")
        elif model_type_choice == '4':  # VGG16
            BATCH_SIZE = 128
            LEARNING_RATE = 0.1
            WEIGHT_DECAY = 5e-4
            MOMENTUM = 0.9
            
            # 学习率阶段设置
            WARMUP_EPOCHS = 5
            INITIAL_STAGE = 60
            FINAL_STAGE = 105
            
            print("Using VGG16 specific parameters:")
            print(f"Batch Size: {BATCH_SIZE}")
            print(f"Epochs: {EPOCHS}")
            print(f"Initial Learning Rate: {LEARNING_RATE}")
            print(f"Weight Decay: {WEIGHT_DECAY}")
        else:  # 标准CNN
            BATCH_SIZE = 1024
            LEARNING_RATE = 0.01
            WEIGHT_DECAY = 0.005
            MOMENTUM = 0.9
            print("Using Standard CNN parameters:")
            print(f"Batch Size: {BATCH_SIZE}")
            print(f"Epochs: {EPOCHS}")
            print(f"Initial Learning Rate: {LEARNING_RATE}")
            print(f"Weight Decay: {WEIGHT_DECAY}")
    else:
        # 测试模式使用默认参数
        BATCH_SIZE = 1024
        LEARNING_RATE = 0.01
        WEIGHT_DECAY = 0.005
        MOMENTUM = 0.9
        WARMUP_EPOCHS = 5
        INITIAL_STAGE = 60
        FINAL_STAGE = 105

    # 如果选择训练，询问是否使用数据增强
    use_augmentation = False
    if choice != 't' and choice != 'T':
        print("Use data augmentation?(y/Y for yes, other for no)")
        aug_choice = input()
        
        if aug_choice == 'y' or aug_choice == 'Y':
            use_augmentation = True
            print("Using data augmentation...")
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.RandomErasing(p=0.5)  # 移到ToTensor之后
            ])
        else:
            print("Not using data augmentation...")
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    else:
        train_transform = test_transform

    # 加载数据集
    full_train_dataset = CustomCIFAR10('./cifar-10-batches-py', train=True, transform=train_transform)
    test_dataset = CustomCIFAR10('./cifar-10-batches-py', train=False, transform=test_transform)

    # 划分训练集和验证集
    val_size = int(len(full_train_dataset) * VALIDATION_SPLIT)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file_prefix = "cifar10_aug_" if use_augmentation else "cifar10_noaug_"
    
    # 选择模型
    if model_type_choice == '2':
        print("Using ResNet model...")
        model = ResNet_CIFAR10(res=True).to(device)
        file_prefix = "resnet_" + file_prefix
    elif model_type_choice == '3':
        print("Using VGG19 model...")
        model = VGG19_CIFAR10().to(device)
        file_prefix = "vgg19_" + file_prefix
    elif model_type_choice == '4':
        print("Using VGG16 model...")
        model = VGG16_CIFAR10().to(device)
        file_prefix = "vgg16_" + file_prefix
    else:
        print("Using standard CNN model...")
        model = CIFAR10Net().to(device)
    
    model_filename = file_prefix + "best.pth"
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # # 使用余弦退火学习率调度
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    # print(f"Using Cosine Annealing LR schedule with T_max={T_MAX}")
    # 使用阶段性学习率调度
    def lr_schedule(epoch):
        if epoch < WARMUP_EPOCHS:
            return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
        elif epoch < INITIAL_STAGE:
            return LEARNING_RATE
        elif epoch < FINAL_STAGE:
            return LEARNING_RATE * 0.1
        else:
            return LEARNING_RATE * 0.01
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    print("Using stage-wise learning rate schedule for VGG19")

    if choice == 't' or choice == 'T':
        # 只进行测试
        print("Test which model?")
        print("1: Standard CNN with augmentation")
        print("2: Standard CNN without augmentation")
        print("3: ResNet with augmentation")
        print("4: ResNet without augmentation")
        print("5: VGG19 with augmentation")
        print("6: VGG19 without augmentation")
        print("7: VGG16 with augmentation")
        print("8: VGG16 without augmentation")
        model_choice = input()
        evaluate_model(model_choice)
    else:
        # 训练模型
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 记录训练过程中的各种指标
        train_losses = []
        val_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(EPOCHS):
            # 训练
            model.train()
            train_loss = 0
            correct = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                train_loss += loss.item() * len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            # 计算平均训练损失和准确率
            train_loss /= len(train_loader.dataset)
            train_accuracy = 100. * correct / len(train_loader.dataset)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({train_accuracy:.2f}%)')

            # 验证
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item() * len(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            scheduler.step()
            val_accuracy = 100. * correct / len(val_loader.dataset)
            
            print(f'Validation set: Average loss: {val_loss:.4f}, '
                  f'Accuracy: {correct}/{len(val_loader.dataset)} ({val_accuracy:.2f}%)')
            
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            # 测试
            test_loss, test_accuracy = test_model(model, device, test_loader, criterion)
            test_accuracies.append(test_accuracy)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_filename)
                print(f"Saved model to {model_filename}")
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
        loss_curve_filename = file_prefix + "loss_curves.png"
        plt.savefig(loss_curve_filename)
        plt.close()
        print(f"保存损失曲线图: {loss_curve_filename}")
        
        # 绘制准确率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Test Accuracy Curves')
        plt.legend()
        plt.grid(True)
        accuracy_curve_filename = file_prefix + "accuracy_curves.png"
        plt.savefig(accuracy_curve_filename)
        plt.close()
        print(f"保存准确率曲线图: {accuracy_curve_filename}")

        # 加载最佳模型进行测试
        print('Loading best model and testing...')
        model.load_state_dict(torch.load(model_filename))
        test_model(model, device, test_loader, criterion)
        print('===Finished Training===')

def test_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def evaluate_model(model_choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = CustomCIFAR10('./cifar-10-batches-py', train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1024)
    
    if model_choice == '1':
        model_filename = "cifar10_aug_best.pth"
        model = CIFAR10Net().to(device)
        model_name = "Standard CNN with augmentation"
    elif model_choice == '2':
        model_filename = "cifar10_noaug_best.pth"
        model = CIFAR10Net().to(device)
        model_name = "Standard CNN without augmentation"
    elif model_choice == '3':
        model_filename = "resnet_cifar10_aug_best.pth"
        model = ResNet_CIFAR10().to(device)
        model_name = "ResNet with augmentation"
    elif model_choice == '4':
        model_filename = "resnet_cifar10_noaug_best.pth"
        model = ResNet_CIFAR10().to(device)
        model_name = "ResNet without augmentation"
    elif model_choice == '5':
        model_filename = "vgg19_cifar10_aug_best.pth"
        model = VGG19_CIFAR10().to(device)
        model_name = "VGG19 with augmentation"
    elif model_choice == '6':
        model_filename = "vgg19_cifar10_noaug_best.pth"
        model = VGG19_CIFAR10().to(device)
        model_name = "VGG19 without augmentation"
    elif model_choice == '7':
        model_filename = "vgg16_cifar10_aug_best.pth"
        model = VGG16_CIFAR10().to(device)
        model_name = "VGG16 with augmentation"
    elif model_choice == '8':
        model_filename = "vgg16_cifar10_noaug_best.pth"
        model = VGG16_CIFAR10().to(device)
        model_name = "VGG16 without augmentation"
    else:
        print("Invalid choice, using Standard CNN with augmentation")
        model_filename = "cifar10_aug_best.pth"
        model = CIFAR10Net().to(device)
        model_name = "Standard CNN with augmentation"
    
    # 加载模型权重
    try:
        print(f"Loading model: {model_filename}")
        model.load_state_dict(torch.load("models/" + model_filename))
        print(f"Model {model_name} loaded successfully")
    except FileNotFoundError:
        print(f"Error: Model file {model_filename} not found")
        return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # 评估模型
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    print("Starting evaluation...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 计算每个类别的准确率
            correct_tensor = pred.eq(target.view_as(pred))
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += correct_tensor[i].item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # 打印总体结果
    print('\nTest results:')
    print(f'Model: {model_name}')
    print(f'Average loss: {test_loss:.4f}')
    print(f'Overall accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    # 打印每个类别的准确率
    print('Class accuracy:')
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'{CLASSES[i]}: {class_acc:.2f}%')
    
    return test_loss, accuracy

if __name__ == '__main__':
    main()

# 实验结果
# 标准CNN:
# 测试集准确率：87.52%（有数据增强，CNN）
# 测试集准确率：82.31%（无数据增强，CNN）
# VGG_MINI: 90.22%
# VGG_16: 94.11%
# VGG_19
# 数据增强提高了约5%的测试准确率

