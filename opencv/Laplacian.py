import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./Python.jpg")
cv2.imshow("img", img)

img_Laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
cv2.imshow("Laplacian", img_Laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()