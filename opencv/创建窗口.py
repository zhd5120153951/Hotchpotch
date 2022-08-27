from asyncio import wait_for
import cv2
import matplotlib as plt
import numpy as np

#创建
cv2.namedWindow("windows", cv2.WINDOW_NORMAL)
cv2.imshow("windows", 0)
cv2.waitKey(5000)
#重置大小
cv2.resizeWindow("windows", 653, 234)
cv2.imshow("windows", 0)
key = cv2.waitKey(0)
if key & 0xff == ord("q"):
    print(ord("q"))  #q字符的ASCII码为113
    cv2.destroyAllWindows()
