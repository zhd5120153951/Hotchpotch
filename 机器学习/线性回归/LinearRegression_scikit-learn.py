'''
@FileName   :LinearRegression_scikit-learn.py
@Description:sk-learn房价预测
@Date       :2022/09/07 16:22:02
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
from __future__ import print_function
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler  #引入归一化的包


def linearRegression():
    print(u"加载数据...\n")
    data = loadtxtAndcsv_data("./机器学习/线性回归/data.txt", ",", np.float64)  #读取数据
    X = np.array(data[:, 0:-1], dtype=np.float64)  # X对应0到倒数第2列
    y = np.array(data[:, -1], dtype=np.float64)  # y对应最后一列

    # 归一化操作
    scaler = StandardScaler()
    scaler.fit(X)
    x_train = scaler.transform(X)
    x_test = scaler.transform(np.array([1650, 3]).reshape(1, -1))

    # 线性模型拟合
    model = linear_model.LinearRegression()
    model.fit(x_train, y)

    #预测结果
    result = model.predict(x_test)
    print("特征系数：", model.coef_)  # Coefficient of the features 决策函数中的特征系数
    print("偏置：", model.intercept_)  # 又名bias偏置,若设置为False，则为0
    print("预测结果：", result[0])  # 预测结果


# 加载txt和csv文件
def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


# 加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)


if __name__ == "__main__":
    linearRegression()