from tkinter import Image
from tkinter.tix import IMAGE
import cv2
import numpy as np
from PIL import ImageDraw
#这些函数不必完全去记，多用多查就可以了
#创建一副背景图，纯黑的
img = np.zeros((480, 640, 3), np.uint8)
#画线
cv2.line(img, (0, 0), (500, 250), (120, 250, 120), 1, 4)
#画矩形
cv2.rectangle(img, (200, 50), (450, 380), (250, 0, 125), 1, 4, 1)
#画圆
cv2.circle(img, (300, 300), 100, (125, 125, 125), 1, 4, 1)
#画椭圆

#画多边形

#绘制文本

cv2.imshow("winna", img)
#cv2.imshow("rectange", img)
cv2.waitKey(0)
cv2.destroyAllWindows()