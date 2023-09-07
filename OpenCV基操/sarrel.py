import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./Python.jpg")
cv2.imshow("img", img)

img_sarrelX = cv2.Scharr(img, cv2.CV_64F, dx=1, dy=0)
cv2.imshow("scharrx", img_sarrelX)
img_sarrelY = cv2.Scharr(img, cv2.CV_64F, dx=0, dy=1)
cv2.imshow("scharry", img_sarrelY)

#备注：dx>=0&&dy>=0&&dx+dy=1---->kisze=-1是等同于sobel算子
#同时，dx是水平方向差分(列)，dy是垂直方向差分(行)
cv2.waitKey(0)
cv2.destroyAllWindows()
