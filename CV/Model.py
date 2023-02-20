'''
@FileName   :Model.py
@Description:模型定义
@Date       :2023/02/16 21:06:20
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import torch
import torch.nn as nn
import numpy as np
"""
基于pytorch框架编写模型训练
1、规律：若一个6维向量的样本： 前三个数大于后三个数--为正样本
                            前三个数等于后三个数--为零样本
                            前三个数小于后三个数--为负样本
2、任务需求：判断任意一个样本的性质？属于哪个类别？
"""


#第一步：利用torch框架，搭建模型
class TorchModel(nn.Module):

    def __init__(self, input_size):
        # 继承父类的构造
        super(TorchModel, self).__init__()
        # 线性层
        self.linear = nn.Linear(input_size, 1)
        # sigmoid归一化函数
        self.activation = torch.sigmoid
        # loss函数采用均方差损失
        self.loss = nn.functional.mse_loss

    # 当输入真实标签--训练时，返回loss值；无真实标签--推理时，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


#纯手工搭建模型
class DiyModel():

    def __init__(self, weight) -> None:
        self.weight = weight

    def forward(self, x, y=None):
        y_pred = np.dot(self.weight, x)
        y_pred = self.diy_sigmoid(y_pred)

        if y is not None:
            return self.diy_mse_loss(y_pred, y)
        else:
            return y_pred

    def diy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def diy_mse_loss(self, y_pred, y_true):
        return np.sum(np.square((y_pred - y_true)) / len(y_pred))

    def calculate_grad(self, y_pred, y_true, x):
        wx = np.dot(self.weight, x)
        sigmoid_wx = self.diy_sigmoid(wx)
        loss = self.diy_mse_loss(sigmoid_wx, y_true)
        #反向过程
        # 均方差函数 (y_pred - y_true) ^ 2 / n 的导数 = 2 * (y_pred - y_true) / n
        grad_loss_sigmoid_wx = 2 / len(x) * (y_pred - y_true)
        # sigmoid函数 y = 1/(1+e^(-x)) 的导数 = y * (1 - y)
        grad_sigmoid_wx_wx = y_pred * (1 - y_pred)
        # wx对w求导 = x
        grad_wx_w = x
        #导数链式相乘
        grad = grad_loss_sigmoid_wx * grad_sigmoid_wx_wx
        grad = np.dot(grad.reshape(len(x), 1), grad_wx_w.reshape(1, len(x)))
        return grad


def diy_sgd(grad, weight, learning_rate):
    return weight - learning_rate * grad


def diy_adam(grad, weight):
    #参数应当放在外面，此处为保持后方代码整洁简单实现一步
    alpha = 1e-3  #学习率
    beta1 = 0.9  #超参数
    beta2 = 0.999  #超参数
    eps = 1e-8  #超参数
    t = 0  #初始化
    mt = 0  #初始化
    vt = 0  #初始化
    #开始计算
    t = t + 1
    gt = grad
    mt = beta1 * mt + (1 - beta1) * gt
    vt = beta2 * vt + (1 - beta2) * gt**2
    mth = mt / (1 - beta1**t)
    vth = vt / (1 - beta2**t)
    weight = weight - (alpha / (np.sqrt(vth) + eps)) * mth
    return weight
