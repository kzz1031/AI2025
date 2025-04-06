# Lab1 Task3 超参数调优、消融实验与ResNet
## BP超参数调优
学习率（3种）、batch size(4种)、参数初始化方式（3种）
```python
# 超参数网格搜索
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [32, 64, 128, 256]
init_methods = ['he', 'xavier', 'standard']
```

He 初始化（He Initialization）
特点：专为 ReLU 激活函数设计，能够有效缓解梯度消失问题。

方法：权重初始化为服从均值为 0、方差为 $( \frac{2}{\text{fanin}} )$ 的正态分布或均匀分布，其中 $fan_{in}$ 是输入神经元的数量。
优势：适用于深层网络，尤其是使用 ReLU 或其变体作为激活函数的网络。

Xavier 初始化（Xavier Initialization）

特点：适用于 Sigmoid 和 Tanh 激活函数，旨在保持前向传播和反向传播时的信号方差一致。

方法：权重初始化为服从均值为 0、方差为 $( \frac{1}{\text{fanin} + \text{fanout}} )$ 的正态分布或均匀分布，其中 fan_in 和 fan_out 分别是输入和输出神经元的数量。

优势：在浅层和中等深度的网络中表现良好。

Standard 初始化（标准初始化）

特点：通常指随机初始化，权重从标准正态分布（均值为 0，方差为 1）中采样。
方法：没有针对特定激活函数或网络深度进行优化。

劣势：在深层网络中可能导致梯度消失或爆炸问题。
> 测试方法：测试3 * 4 * 3种方法的组合，测试10轮训练之后在验证集上的表现（较为不严谨，但是省时间，训完整一轮还是要好久...）

```python
for lr in learning_rates:
    for bs in batch_sizes:
        for init in init_methods:
            print(f'\nTesting lr={lr}, batch_size={bs}, init={init}')
            
            # 初始化模型
            network = NeuralNetwork(init_method=init)
            n_batches = X_train.shape[0] // bs
            
            # 训练5个epoch来快速评估参数组合
            for epoch in range(10):
                total_loss = 0
                for i in range(n_batches):
                    batch_x = X_train[i * bs:(i + 1) * bs]
                    batch_y = y_train_onehot[i * bs:(i + 1) * bs]
                    total_loss += network.train_step(batch_x, batch_y, lr)
                
                val_acc = network.evaluate(X_val, y_val)
                print(f'Epoch {epoch + 1}: Loss {total_loss/n_batches:.4f}, Val Acc {val_acc:.4f}')
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': bs,
                        'init_method': init
                    }
```
最后找到的最优方法如下
Best parameters found:
Learning rate: 0.1
Batch size: 32
Init method: he

在该方法下最后测试集上准确率：98.07%
## MNIST(ResNet)
训练结果：
![](pic/minist_accuracy_curves.png)
![](pic/loss_curves.png)
准确率：99.71%
## cifar-10(ResNet)
![](pic/cifar10_shuffle__accuracy_curve1.png)
![](pic/cifar10_shuffle__loss_curve1.png)
准确率：95.70%