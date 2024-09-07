'''
@FileName   :fusion.py
@Description:
@Date       :2021/09/18 10:18:23
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import numpy as np


def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# fg = cv2.imread('./Images/f1.jpg', cv2.INTER_AREA)
# bg = cv2.imread('./Images/f2.jpg', cv2.INTER_AREA)
fg = cv_imread('./OpenCV视觉项目/基于OpenCV的图像融合/f2.jpg')
bg = cv_imread('./OpenCV视觉项目/基于OpenCV的图像融合/f1.jpg')

print(bg.shape)
print(fg.shape)

dim = (500, 300)
resized_fg = cv2.resize(fg, dim, interpolation=cv2.INTER_AREA)
resized_bg = cv2.resize(bg, dim, interpolation=cv2.INTER_AREA)

print(resized_fg.shape)
print(resized_bg.shape)

blend = cv2.addWeighted(resized_bg, 0.2, resized_fg, 0.8, 0.0)
cv2.imshow('ret', blend)
cv2.waitKey(0)
