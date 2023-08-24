'''
@FileName   :多线程读取rtsp.py
@Description:
@Date       :2023/08/24 17:01:33
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import queue
import time
import threading

q = queue.Queue(maxsize=10)


def ReadFrame():
    print("start Reveive")
    cap = cv2.VideoCapture("rtsp://admin:jiankong123@192.168.23.10")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def ProcessFrame():
    print("Start Displaying")
    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    th1 = threading.Thread(target=ReadFrame)
    th2 = threading.Thread(target=ProcessFrame)
    th1.start()
    th2.start()