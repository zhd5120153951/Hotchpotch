'''
@FileName   :基于深度学习的超分辨处理.py
@Description:
@Date       :2022/09/04 16:33:37
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
from cv2 import dnn_superres  #需要安装opencv-contribu包，卸载当前opencv-python

# 创建一个超分辨模型对象
sr = dnn_superres.DnnSuperResImpl_create()

img = cv2.imread('./OpenCV视觉项目/基于深度学习的超分辨处理/1.jpg')
path = './OpenCV视觉项目/基于深度学习的超分辨处理/EDSR_x3.pd'  #x2,x3,x4
sr.readMoodel(path)

sr.setModel('edsr', 3)
ret = sr.upsample(img)
cv2.imshow('upscale', ret)
