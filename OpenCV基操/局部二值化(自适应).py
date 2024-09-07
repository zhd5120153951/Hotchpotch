import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Python.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.adaptiveThreshold(img, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
cv2.imshow("局部二值化", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
