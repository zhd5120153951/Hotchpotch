import cv2
import numpy as np
import matplotlib as plt

img1 = cv2.imread("test7.jpg")
img2 = cv2.imread("test8.jpg")
#缩放变换
img3 = cv2.resize(img1, (720, 640), 0.5, 1.5, cv2.INTER_LINEAR)
cv2.imshow("winna", img3)
#图像翻转
#img4 = img2[::-1, ::-1]法①
img4 = cv2.flip(img3, -1)
cv2.imshow("on", img4)
#图像旋转
img5 = cv2.rotate(img3, 90)
cv2.imshow("ok", img5)
#还有一个仿射变换
cv2.waitKey(0)