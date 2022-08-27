from operator import ipow
from pickle import FALSE
import cv2
import numpy as np

cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("video", 640, 480)

#一直在读取了
cap = cv2.VideoCapture(0)  #如果是0表示用本地摄像头,如果是路径则是本地视频
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if (key & 0xff == ord("q")):
        break
cap.release()
cv2.destroyAllWindows()