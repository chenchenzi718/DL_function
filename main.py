from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt


def target_func(input_data):
    return np.sin(input_data) + np.exp(-input_data)

# 设置跑的平台是CPU还是GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)
print(torch.cuda.is_available())
from torch.backends import cudnn
print(cudnn.is_available())

# 用flag来做测试集测试
test_flag = 1
# 用flag来做验证集测试
validate_flag = 0
# 用new_dataset_test_flag研究这个模型在另外选取的测试集数据点上的效果
new_dataset_test_flag = 1

# 首先产生[0,4pi]中的数据点
x = np.linspace(0, 4*np.pi, 1000)
if validate_flag:
    np.random.seed(0)
np.random.shuffle(x)
y = target_func(x)

# 将x，y变成列向量的形式
change_x = np.expand_dims(x, axis=1)
change_y = np.expand_dims(y, axis=1)

# 使用TensorDataset将x,y做打包;分成三个部分作为训练集，验证集与测试集
dataset_train = TensorDataset(torch.tensor(change_x[0:600, ...], dtype=torch.float),
                              torch.tensor(change_y[0:600, ...], dtype=torch.float))
dataset_validate = TensorDataset(torch.tensor(change_x[600:800, ...], dtype=torch.float),
                                 torch.tensor(change_y[600:800, ...], dtype=torch.float))
dataset_test = TensorDataset(torch.tensor(change_x[800:1000, ...], dtype=torch.float),
                                 torch.tensor(change_y[800:1000, ...], dtype=torch.float))

# 将三个dataset做一次batch打包
dataloader_train = DataLoader(dataset_train, batch_size=100, shuffle=True)
dataloader_validate = DataLoader(dataset_validate, batch_size=100, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=100, shuffle=True)

# 建立一个网络的类
class MyNet(nn.Module):
    def __init__(self, hidden_num, hidden_width):
        super(MyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.hidden_layer_num = hidden_num
        self.hidden_width = hidden_width
        self.fc1 = nn.Linear(in_features=1, out_features=self.hidden_width)
        self.activate_func = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_width, 1)
        self.hidden_layer = nn.Linear(self.hidden_width, self.hidden_width)


    def forward(self, _input):
        out = self.fc1(_input)
        out = self.activate_func(out)
        for i in range(self.hidden_layer_num-1):
            out = self.hidden_layer(out)
            out = self.activate_func(out)
        out = self.fc2(out)
        return out

# 设置参数来调整网络的隐藏层宽度与隐藏层深度
hidden_layer_num = 4
hidden_layer_width = 65

# 设置随机数种子，为了复现实验结果，同时方便验证集上对比试验
if validate_flag:
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
fnn_net = MyNet(hidden_layer_num, hidden_layer_width).to(device)

learning_rate = 0.001
epoch_num = 1500

# 定义优化器和损失函数
optimizer = torch.optim.Adam(MyNet.parameters(fnn_net), lr=learning_rate)
loss_func = nn.MSELoss()

# 训练次数由自己决定为epoch_num
for epoch in range(epoch_num):
    loss = None
    for batch_x, batch_y in dataloader_train:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predict_y = fnn_net(batch_x)
        loss = loss_func(predict_y, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 每100次的时候打印一次日志
    if (epoch+1) % 100 == 0:
        print("step: {0} , loss: {1}".format(epoch+1, loss.item()))

# 使用验证集来检验不同的超参数对结果的影响
if validate_flag:
    loss = 0.0
    for batch_x, batch_y in dataloader_validate:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predict_y = fnn_net(batch_x)
        loss += loss_func(predict_y, batch_y).item()
    # 计算出在验证集上的误差大小
    loss = (loss * dataloader_test.batch_size) / len(dataset_test)
    print("loss on validate set is {0}".format(loss))


# 使用训练好的模型对测试集进行测试，并与结果进行对比得出loss
if test_flag:
    loss = 0.0
    for batch_x, batch_y in dataloader_test:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predict_y = fnn_net(batch_x)
        loss += loss_func(predict_y, batch_y).item()
    # 计算出在测试集上的误差大小
    loss = (loss * dataloader_test.batch_size)/len(dataset_test)
    print("loss on test set is {0}".format(loss))


# 绘图展示在另外挑选的更大的测试集上预测值和真实值之间的差异
if new_dataset_test_flag:
    x = np.linspace(0, 4*np.pi, 670)
    y = target_func(x)
    change_x = np.expand_dims(x, axis=1)
    change_y = np.expand_dims(y, axis=1)
    predict_y = fnn_net(torch.tensor(change_x, dtype=torch.float).to(device))
    loss = loss_func(predict_y, torch.tensor(change_y, dtype=torch.float).to(device)).item()
    print("loss on new dataset test set is {0}".format(loss))
    plt.plot(x, y, label="real value")
    plt.plot(x, predict_y.cpu().detach().numpy(), label="predict value")
    plt.title("sin(x)+exp(-x) function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
