import cv2
import numpy as np

img = cv2.imread("pic.png")
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=2)  #迭代次数设置为一
cv2.imshow("erosion", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
