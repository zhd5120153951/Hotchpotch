import cv2
import numpy as np
import matplotlib as plt

img1 = cv2.imread("test7.jpg")
img2 = cv2.imread("test8.jpg")
img3 = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)
cv2.imshow("winna", img3)
cv2.waitKey(0)