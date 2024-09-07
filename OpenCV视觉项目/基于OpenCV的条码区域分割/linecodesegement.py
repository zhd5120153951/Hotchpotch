'''
@FileName   :linecodesegement.py
@Description:
@Date       :2021/09/19 10:53:29
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取图像
img = cv2.imdecode(np.fromfile('./OpenCV视觉项目/基于OpenCV的条码区域分割/linecode.jpg', np.uint8), -1)
#灰度处理
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('src_gray', img)

img_out = cv2.imdecode(np.fromfile('./OpenCV视觉项目/基于OpenCV的条码区域分割/linecode.jpg', np.uint8), -1)

#调整大小--可不要
scale = 800.0 / img.shape[1]
img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

#黑帽运算--灰度图
kernel = np.ones((1, 3), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, anchor=(1, 0))
# cv2.imshow('blackhat', img)

#阈值处理>10置为255--白
thresh, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
# cv2.imshow('thresh', img)

#膨胀
kernel = np.ones((1, 5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=2)
# cv2.imshow('dilate', img)
# 闭运算
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)
# cv2.imshow('close', img)
#开运算(开闭运算作用:去除孤立的像素点--噪声)
kernel = np.ones((21, 35), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
# cv2.imshow('open', img)
#检测连通区域并筛选大于阈值的区域(条码区域)
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
unscale = 1.0 / scale
if contours != None:
    for contour in contours:
        if cv2.contourArea(contour) <= 2000:
            continue
        rect = cv2.minAreaRect(contour)
        #rect--是每个区域的四个点--最小区域
        rect = \
            ((int(rect[0][0] * unscale), \
              int(rect[0][1] * unscale)),\
             (int(rect[1][0] * unscale), \
              int(rect[1][1] * unscale)), rect[2])

        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(img_out, [box], 0, (0, 255, 0), thickness=1)

# plt.imshow(img_out)
cv2.imshow('ret', img_out)
cv2.waitKey(0)