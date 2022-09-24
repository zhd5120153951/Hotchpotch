import cv2
import numpy as np

img = cv2.imread("pic.png")
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=2)  #迭代次数设置为一
cv2.imshow("erosion", erosion)
#对已经腐蚀过的图像进行操作
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(erosion, kernel, iterations=2)
cv2.imshow('dilate', dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
