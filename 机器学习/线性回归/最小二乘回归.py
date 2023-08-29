'''
@FileName   :最小二乘回归.py
@Description:最小二乘回归
@Date       :2022/08/22 21:54:03
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
@PS         :
'''
import numpy as np


class LR_LS():
    def __init__(self):
        self.w = None

    def train(self, X, y):
        # 最小二乘法矩阵求解
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        y_pred = X.dot(self.w)
        return y_pred


if __name__ == "__main__":
    #随机数
    np.random.seed(1234)
    x = np.random.rand(500, 3)
    print(x.shape)

    #构建映射关系，模拟真实的数据待预测值,映射关系为y = 4.2 + 5.7*x1 + 10.8*x2，可自行设置值进行尝试
    y = x.dot(np.array([4.2, 5.7, 10.8]))  #矩阵乘法--x.dot(y)==np.dot(x,y)
    print(y.shape)

    lr_ls = LR_LS()
    lr_ls.train(x, y)
    print("估计的参数值：%s" % (lr_ls.w))  #训练后得到的权重
    x_test = np.array([2, 4, 5]).reshape(1, -1)
    print("预测值为: %s" % (lr_ls.predict(x_test)))
