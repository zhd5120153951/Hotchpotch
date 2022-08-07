import cv2
import matplotlib as plt
import numpy as np

img = cv2.imread("./1.jpg")

cv2.imshow("person", img)

cv2.waitKey(0)
