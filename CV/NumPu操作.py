from cmath import sqrt
import torch
import numpy as np

#基本操作
x = np.array([[1, 2, 3], [4, 5, 6]])
#维度
print(x.ndim)
#形状
print(x.shape)
#元素个数
print(x.size)
#所有元素求和
print(np.sum((x)))
#列方向对应求和==向量---沿x
print(np.sum(x, axis=0))
#行方向对应求和==向量---沿y
print(np.sum(x, axis=1))
#重构形状
print(np.reshape(x, (3, 2)))
#开方每个元素
print(np.sqrt(x))
#e乘方
print(np.exp(x))
#转置
print(np.transpose(x))
#平滑展开为一位
print(x.flatten())

#把矩阵x转为张良
x = torch.FloatTensor(x)
print(x.shape)
print(torch.exp(x))
print(torch.sum(x, dim=0))
print(torch.sum(x, dim=1))
print(x.transpose(1, 0))
print(x.flatten())
