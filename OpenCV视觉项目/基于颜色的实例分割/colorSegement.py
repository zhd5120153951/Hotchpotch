'''
@FileName   :colorSegement.py
@Description:
@Date       :2022/09/07 17:35:28
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def OnTracbarChange1(value):
    global s1
    s1 = value


def OnTracbarChange2(value):
    global s2
    s2 = value


def OnTracbarChange3(value):
    global s3
    s3 = value


if __name__ == '__main__':
    #返回Image对象--区别于cv2.imread()
    img = Image.open('./OpenCV视觉项目/基于颜色的实例分割/bird.jpg')
    #转换成cv2.imread()格式
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #创建一个窗口
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #调整窗口大小
    cv2.resizeWindow('image', 800, 600)
    cv2.createTrackbar('slider_1', 'image', 0, 255, OnTracbarChange1)
    cv2.createTrackbar('slider_2', 'image', 0, 255, OnTracbarChange2)
    cv2.createTrackbar('slider_3', 'image', 0, 255, OnTracbarChange3)
    s1, s2, s3 = 0, 0, 0
    cv2.imshow('image', img)

    while True:
        #模糊操作--opencv内置4种模糊操作
        blur1 = cv2.blur(img, (3, 3))
        blur2 = cv2.medianBlur(blur1, 3)
        blur3 = cv2.GaussianBlur(blur2, (3, 3), 0)
        blur4 = cv2.bilateralFilter(blur3, 9, 75, 75)
        cv2.imshow('blur4', blur4)
        # BGR->HSV(因为颜色像素不能很好区分目标和背景，而亮度，饱和度、色度可以)
        hsv = cv2.cvtColor(blur4, cv2.COLOR_BGR2HSV)

        # 颜色分割--阈值分割--把感兴趣区域置为1，区域为0
        low_blue = np.array([55, 0, 0])
        high_blue = np.array([s1, s2, s3])
        #掩码--蒙板操作
        mask = cv2.inRange(hsv, low_blue, high_blue)

        cv2.imshow('mask', mask)

        #add操作--位与
        ret = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('ret', ret)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break