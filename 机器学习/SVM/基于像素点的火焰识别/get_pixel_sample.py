'''
@FileName   :get_pixel_sample.py
@Description:
@Date       :2023/08/24 16:14:34
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import numpy as np
import cv2
import time
import pandas as pd
import joblib
import argparse
import os


def click_event(event, x, y, flags, img):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ',', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv2.putText(img, strXY, (x, y), font, 1, (255, 255, 0), 1)
        cv2.imshow('image', img)
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ', ' + str(green) + ', ' + str(red)
        print(f"{strBGR}")
        with open("images/data/nofire.txt", "a+") as f:
            f.writelines(strBGR)
            f.writelines("\n")


def get_points(pic_path=''):
    cv2.namedWindow('image', 0)
    img = cv2.imread(pic_path)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event, img)
    cv2.waitKey(0)


if __name__ == "__main__":
    pic_path = './images/nofire/'
    files = os.listdir(pic_path)
    i = 0
    for file in files:
        print(file, i)
        get_points(pic_path + file)
        i = i + 1