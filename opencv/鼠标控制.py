from pickletools import uint8
import cv2
import numpy as np


def mouse_callback(event, x, y, flags, userdata):
    print(event, x, y, flags, userdata)
    if event == 2:
        cv2.destroyAllWindows()


cv2.namedWindow("mouse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mouse", 640, 480)

cv2.setMouseCallback("mouse", mouse_callback, "123")

#自动生成一张全黑图片
img = np.zeros((480, 640, 3), np.uint8)
while True:
    cv2.imshow("mouse", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()