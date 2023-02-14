#coding:utf8

import torch
import torch.nn as nn
import numpy as np
"""
numpy手动实现模拟一个线性层
"""


#搭建一个2层的神经网络模型
#每层都是线性层
class TorchModel(nn.Module):
    '''
    这就是在定义模型结构
    '''

    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)

    #这是前向传播---等价于在计算y=w*x+b
    def forward(self, x):
        hidden = self.layer1(x)  #shape: (batch_size, input_size) -> (batch_size, hidden_size1)
        y_pred = self.layer2(hidden)  #shape: (batch_size, hidden_size1) -> (batch_size, hidden_size2)
        return y_pred


#自定义模型
class DiyModel:
    #同理：定义模型结构---自定义和框架的区别就在这里：框架已经封装好了包函数--我们只需要确定输入、每层的形状，随机权重就可以了
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    #y1 = w1*x+b---> y=y1*w2+b2--->y输出
    def forward(self, x):
        #第一层计算
        hidden = np.dot(x, self.w1.T) + self.b1  #1*5
        #第二层计算
        y_pred = np.dot(hidden, self.w2.T) + self.b2  #1*2
        return y_pred


#随便准备一个网络输入
x = np.array([[3.1, 1.3, 1.2], [2.1, 1.3, 13]])
#建立torch模型
torch_model = TorchModel(3, 5, 2)
print(torch_model.state_dict())
print("-----------")
#打印模型权重，权重为随机初始化
#第一层的初始化
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_b1 = torch_model.state_dict()["layer1.bias"].numpy()
#第二层的初始化
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_b2 = torch_model.state_dict()["layer2.bias"].numpy()
print(torch_model_w1, "torch w1 权重")
print(torch_model_b1, "torch b1 权重")
print("-----------")
print(torch_model_w2, "torch w2 权重")
print(torch_model_b2, "torch b2 权重")
print("-----------")
#使用torch模型做预测
torch_x = torch.FloatTensor(x)
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred)
#把torch模型权重拿过来自己实现计算过程
diy_model = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
#用自己的模型来预测
y_pred_diy = diy_model.forward(np.array(x))
print("diy模型预测结果：", y_pred_diy)
