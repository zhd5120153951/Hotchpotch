from operator import ipow
from pickle import FALSE
import cv2
import numpy as np

cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("video", 640, 480)

#一直在读取了
cap = cv2.VideoCapture(0)  #如果是0表示用本地摄像头,如果是路径则是本地视频
#要采集的视频格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#开始读取
vw = cv2.VideoWriter("./output.mp4", fourcc, 30, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        print("can't recive frame")
        break
    #开始写入缓存
    vw.write(frame)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if (key & 0xff == ord("q")):
        break

cap.release()
#写入磁盘和释放资源
vw.release()
cv2.destroyAllWindows()