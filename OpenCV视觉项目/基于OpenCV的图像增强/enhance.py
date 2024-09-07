'''
@FileName   :enhance.py
@Description:
@Date       :2022/09/22 13:29:01
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imdecode(np.fromfile('./OpenCV视觉项目/基于OpenCV的图像增强/low.jpg', np.uint8), -1)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

images = np.concatenate((img, img), axis=1)
cv2.imshow('origin', images)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray_img = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
# 直方图
hist = cv2.calcHist(gray_img, [0], None, [256], [0, 256])
plt.subplot(121)
plt.title("Images")
plt.xlabel("bins")
plt.ylabel("No of pixels")
plt.plot(hist)
plt.show()
# cv2.imshow('hist', hist)

#均衡化
gray_img_eqhist = cv2.equalizeHist(gray_img)
hist = cv2.calcHist(gray_img_eqhist, [0], None, [256], [0, 256])
plt.subplot(121)
plt.plot(hist)
plt.show()
cv2.imshow('eqhist', gray_img_eqhist)
cv2.waitKey(0)
cv2.destroyAllWindows()

clahe = cv2.createCLAHE(clipLimit=40)
gray_img_clahe = clahe.apply(gray_img_eqhist)
cv2.imshow('clahe', gray_img_clahe)
cv2.waitKey(0)
cv2.destroyAllWindows()

#自适应阈值--常用全局阈值cv2.THRESHOLD_BINATY
thresh1 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1)
thresh2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1)
thresh3 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
thresh4 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)

cv2.imshow('thresh1', thresh1)
cv2.imshow('thresh2', thresh2)
cv2.imshow('thresh3', thresh3)
cv2.imshow('thresh4', thresh4)

cv2.waitKey(0)
cv2.destroyAllWindows()

#OTSU二值化
ret, th1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('otsu', th1)
cv2.waitKey(0)
cv2.destroyAllWindows()