import cv2
import os

file_list = []
img_list = os.listdir()
img = cv2.imread('D:\\Zengh\\428_.jpg')
# print(img.shape)
# cv2.imshow('orgin', img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.merge((img, img, img))
# img = cv2.convertScaleAbs(img)
# print(img.shape)
# cv2.imshow('merge', img)

cv2.imwrite('D:\\Zengh\\gray.jpg', img)
