'''
@FileName   :demo1.py
@Description:
@Date       :2022/09/22 11:10:53
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import numpy as np


#掩码操作
def mask_operate(img):
    h = img.shape[0]
    polygons = np.array([[(0, h), (800, h), (250, 100)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    mask_img = cv2.bitwise_and(img, mask)
    return mask_img


#读图
img = cv2.imdecode(np.fromfile('./OpenCV视觉项目/基于OpenCV的实用图像操作/line.jpg', np.uint8), -1)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('origin', img)

#灰度
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#阈值处理
(thresh, thresh_img) = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
#高斯模糊--不做这一步会有很多噪声(canny)
gaussian_img = cv2.GaussianBlur(thresh_img, (3, 3), 3)
cv2.imshow('gaussian', gaussian_img)
#边缘检测
canny_img = cv2.Canny(thresh_img, 180, 255)
cv2.imshow('canny', canny_img)

canny_img = mask_operate(canny_img)

#获取边缘区域
lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 30)
print(lines.shape)
for line in lines:
    # print(line)
    # x1, y1, x2, y2 = line[0]
    cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
cv2.imshow('line', img)

cv2.waitKey(0)