'''
@FileName   :diff_frame_3_add.py
@Description:三帧差分求和
@Date       :2022/11/03 09:03:13
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
from matplotlib import contour
import numpy as np

# 没有背景减除--因为背景一般不会动--帧差后为0--但也会影响效果


def diff_3frame_add(videopath):
    cap = cv2.VideoCapture(videopath)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    one_frame = np.zeros((height, width), dtype=np.uint8)
    two_frame = np.zeros((height, width), dtype=np.uint8)
    three_frame = np.zeros((height, width), dtype=np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray

        abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
        _, thresh1 = cv2.threshold(
            abs1, 40, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0
        abs2 = cv2.absdiff(two_frame, three_frame)
        _, thresh2 = cv2.threshold(abs2, 40, 255, cv2.THRESH_BINARY)

        binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erode = cv2.erode(binary, kernel)  # 腐蚀
        dilate = cv2.dilate(erode, kernel)  # 膨胀
        dilate = cv2.dilate(dilate, kernel)  # 膨胀,可以膨胀两次,可以更好地去掉孤立点

        contours, _ = cv2.findContours(
            dilate.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        for contour in contours:
            if 10 < cv2.contourArea(contour) < 500:
                x, y, w, h = cv2.boundingRect(contour)  # 找方框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("binary", binary)
        cv2.imshow("dilate", dilate)
        cv2.imshow("frame", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# 增加背景减除--边缘检测挺好的

def diff_frame_nobkg(videopath):
    cap = cv2.VideoCapture(videopath)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fbkg = cv2.createBackgroundSubtractorMOG2()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame1 = np.zeros((640, 480))
    out = cv2.VideoWriter('test.avi', fourcc, 10, (640, 480))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('ori', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        fmask = fbkg.apply(frame)

        mask = cv2.morphologyEx(fmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('mask', mask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        contours, _ = cv2.findContours(
            fmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if 1000 < cv2.contourArea(contour) < 2000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            out.write(frame)
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # diff_3frame_add(0)
    diff_frame_nobkg(0)
