'''
@FileName   :operate.py
@Description:
@Date       :2022/09/22 09:53:44
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np

# cv2.imread()出现汉字编码有问题时,用imdecode()
img = cv2.imdecode(np.fromfile('./OpenCV视觉项目/基于OpenCV的实用图像操作/line.jpg', np.uint8), -1)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
print(img.shape)

# 转为灰度--3个维度-->2个维度
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray_img)
# plt.imshow(gray_img)
# plt.show()
print(gray_img.shape)

# 阈值功能(把大于阈值的像素值变成255,其余为0)--对灰度图--变成二值图
(thresh, blackAndWhiteImage) = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh20', blackAndWhiteImage)

(thresh, blackAndWhiteImage) = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh80', blackAndWhiteImage)

(thresh, blackAndWhiteImage) = cv2.threshold(gray_img, 16, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh160', blackAndWhiteImage)

(thresh, blackAndWhiteImage) = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh200', blackAndWhiteImage)

#模糊处理--消除噪点
blur_img = cv2.GaussianBlur(gray_img, (3, 3), 5)
cv2.imshow('blur', blur_img)

#图像旋转
(h, w) = img.shape[:2]
center = (w / 2, h / 2)
M = cv2.getRotationMatrix2D(center, 13, scale=1.0)  #旋转不缩放
rotated_img = cv2.warpAffine(gray_img, M, (w, h))
cv2.imshow('ratated', rotated_img)

cv2.waitKey(0)
