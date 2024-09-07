import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Python.jpg")
cv2.imshow("img", img)

img_sobel = cv2.Sobel(img, -1, dx=1, dy=1, ksize=3)
cv2.imshow("sobel", img_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()
