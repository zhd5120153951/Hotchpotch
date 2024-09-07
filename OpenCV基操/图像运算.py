# 图像运算中，只有加法用得较多
import cv2
import numpy as np
import matplotlib as plt

img1 = cv2.imread("test7.jpg")
img2 = cv2.imread("test8.jpg")
img3 = cv2.add(img1, img2)
cv2.imshow("winna", img3)
cv2.waitKey(0)
