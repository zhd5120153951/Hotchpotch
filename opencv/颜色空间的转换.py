import cv2
import numpy as np
import matplotlib as plt


def callback():
    pass


cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.resizeWindow("win", 640, 480)

img = cv2.imread("./1.jpg")
color_space = [cv2.COLOR_BGR2BGRA, cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2YUV]
cv2.createTrackbar("bar", "win", 0, 4, callback)
while True:
    index = cv2.getTrackbarPos("bar", "win")
    cvt_img = cv2.cvtColor(img, color_space[index])
    cv2.imshow("win", cvt_img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
