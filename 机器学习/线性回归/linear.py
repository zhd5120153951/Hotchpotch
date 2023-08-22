'''
@FileName   :linear.py
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

#数据生成
import numpy as np
#随机数
np.random.seed(1234)
x = np.random.rand(500, 3)
print(x.shape)
print(x)
# print(x[3, 2])
# print(x[2, 2])

#构建映射关系，模拟真实的数据待预测值,映射关系为y = 4.2 + 5.7*x1 + 10.8*x2，可自行设置值进行尝试
y = x.dot(np.array([4.2, 5.7, 10.8]))  #矩阵乘法--x.dot(y)==np.dot(x,y)
print(y.shape)
print(y)

# np.array([4.2,5.7,10.8])--(3,)--既不是行向量也不是列向量，只是有3个元素，是一个维度为3的数组

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# %matplotlib inline

# 调用模型
lr = LinearRegression(fit_intercept=True)
# 训练模型
lr.fit(x, y)
print("估计的参数值为：%s" % (lr.coef_))
# 计算R平方
print('R2:%s' % (lr.score(x, y)))
# 任意设定变量，预测目标值
x_test = np.array([2, 4, 5]).reshape(1, -1)  # 定义一个行向量
y_hat = lr.predict(x_test)
print("预测值为: %s" % (y_hat))
