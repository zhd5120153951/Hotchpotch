'''
@FileName   :检测马赛克图像.py
@Description:
@Date       :2024/11/20 14:10:38
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

from PIL import Image
import numpy as np
import math
import warnings

high_thresh = 200  # 高阈值
low_thresh = 40  # 低阈值
warnings.filterwarnings('ignore')
demo = Image.open('CAPTCHA.jpg')
im = np.array(demo.convert('L'))  # 灰度变换矩阵
print(f"变换后的图像形状:{im.shape}")
print(f"变换后的图像数据类型:{im.dtype}")
# 一般size指宽高(width,height);shape指行列(height,width)
height = im.shape[0]
width = im.shape[1]
gm = [[0 for i in range(width)]for j in range(height)]  # 梯度强度
gx = [[0 for i in range(width)]for j in range(height)]  # 梯度x
gy = [[0 for i in range(width)]for j in range(height)]  # 梯度y

theta = 0  # 梯度方向角初始化--[0,360]
dirr = [[0 for i in range(width)]for j in range(height)]  # 0,1,2,3四个方向的判定值
highorlow = [[0 for i in range(width)]for j in range(height)]  # 强边缘,弱边缘,忽略判定值
rm = np.array([[0 for i in range(width)]for j in range(height)])  # 输出矩阵
# 高斯平滑滤波
for i in range(1, height-1, 1):
    for j in range(1, width-1, 1):
        rm[i][j] = im[i-1][j-1]*0.0924+im[i-1][j] * \
            0.1192+im[i-1][j+1]*0.0924+im[i][j-1]*0.1192
