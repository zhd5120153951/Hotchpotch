import cv2
import numpy as np
import matplotlib as plt


def callback():
    pass


#创建窗口
cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.resizeWindow("win", 640, 480)
#读取图片
img = cv2.imread("./1.jpg")
#定义颜色空间列表
color_space = [cv2.COLOR_BGR2BGRA, cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2YUV]
#创建trackbar
cv2.createTrackbar("bar", "win", 0, 4, callback)
while True:
    index = cv2.getTrackbarPos("bar", "win")
    cvt_img = cv2.cvtColor(img, color_space[index])
    cv2.imshow("win", cvt_img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
