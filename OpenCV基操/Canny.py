import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./Python.jpg")
cv2.imshow("img", img)


def nothing(x):
    pass


cv2.namedWindow("res")
cv2.createTrackbar("min", "res", 0, 25, nothing)
cv2.createTrackbar("max", "res", 0, 25, nothing)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break
    maxvalue = cv2.getTrackbarPos("max", "res")
    minvalue = cv2.getTrackbarPos("min", "res")
    img_canny = cv2.Canny(img, 10 * minvalue, 10 * maxvalue)
    cv2.imshow("res", img_canny)

cv2.destroyAllWindows()
#备注：canny算子可以调整边缘，由阈值(上下限)