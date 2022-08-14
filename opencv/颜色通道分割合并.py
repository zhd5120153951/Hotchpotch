import cv2
import numpy as np

img = np.zeros((480, 640, 3), np.uint8)
b, g, r = cv2.split(img)  #三个通道
b[10:100, 10:200] = 255
g[20:150, 20:300] = 120
r[30:200, 30:400] = 255
img2 = cv2.merge((b, g, r))
cv2.imshow("b", img)
cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
cv2.imshow("merge", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
