import cv2
import numpy as np
import matplotlib.pyplot as plt
#读取图片
img = cv2.imread("1.jpg")
#取得卷积核
#cv2.filter2D()--如果需要自己创建卷积核的用这个函数
#均值滤波
img_mean = cv2.blur(img, (3, 3))
cv2.imshow("mean", img_mean)
#img_box = cv2.boxFilter(img, 1, (3, 3))
#cv2.imshow("box", img_box)
img_guassian = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow("guassiian", img_guassian)

img_median = cv2.medianBlur(img, 3)
cv2.imshow("median", img_median)

img_bilater = cv2.bilateralFilter(img, 3, 75, 75)
cv2.imshow("bilater", img_bilater)

cv2.waitKey(0)
cv2.destroyAllWindows()