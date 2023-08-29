'''
@FileName   :线性回归.py
@Description:线性回归
@Date       :2022/08/22 21:03:19
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
@PS         :
'''

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class LinearRegess():
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def train(self):
        lr = LinearRegression(fit_intercept=True)
        lr.fit(self.x, self.y)
        print("估计的参数值为：%s" % (lr.coef_))
        # 计算R平方
        print('R2:%s' % (lr.score(self.x, self.y)))

        x_test = np.array([2, 4, 5]).reshape(1, -1)  # 把数组变成不同的维度向量
        print(x_test.shape)
        y_hat = lr.predict(x_test)  #输入必须是矩阵--数组不行
        print("预测值为: %s" % (y_hat))


if __name__ == '__main__':
    #随机生成500组数据
    np.random.seed(1234)
    x = np.random.rand(500, 3)
    y = x.dot(np.array([4.2, 5.7, 10.8]))  #初始化三个权重
    # y = x.dot(np.array(np.random.rand(3)))

    linear = LinearRegess(x, y)
    linear.train()
'''
不用类实现
'''
# #随机数
# np.random.seed(1234)
# x = np.random.rand(500, 3)  #生成二维数组[500,3]-->[[x,y,z],...,[x500,y500,z500]]
# print(x.shape)

# #构建映射关系，模拟真实的数据待预测值,映射关系为y = 4.2 + 5.7*x1 + 10.8*x2，可自行设置值进行尝试
# y = x.dot(np.array([4.2, 5.7, 10.8]))  #矩阵乘法--x.dot(y)==np.dot(x,y)
# print(y.shape)
# # print(y)
# # np.array([4.2,5.7,10.8])--(3,)--既不是行向量也不是列向量，是有3个元素的一个维度为3的数组

# # 调用模型
# lr = LinearRegression(fit_intercept=True)
# # 训练模型
# lr.fit(x, y)
# print("估计的参数值为：%s" % (lr.coef_))
# # 计算R平方
# print('R2:%s' % (lr.score(x, y)))
# # 任意设定变量--输入，预测目标值
# x_test = np.array([2, 4, 5]).reshape(1, -1)  # 把数组变成不同的维度向量
# print(x_test.shape)
# y_hat = lr.predict(x_test)  #输入必须是矩阵--数组不行
# print("预测值为: %s" % (y_hat))
