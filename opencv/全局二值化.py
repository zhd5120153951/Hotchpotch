import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test8.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("全局二值化", img)
print("ret:{0}".format(ret))
cv2.waitKey(0)
cv2.destroyAllWindows()
