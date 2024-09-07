'''
@FileName   :conver.py
@Description:
@Date       :2022/09/11 15:43:58
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
import numpy as np


#读取图像,解决imread()中文路径读取出错问题
def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


img_path = './OpenCV应用项目/基于OpenCV的图像覆盖/15.jpg'
img = cv_imread(img_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img)

img[50:150, 50:150] = [36, 72, 108]
cv2.imshow('ret', img)

# resize_img = cv2.resize(img, dsize=(100, 100))
resize_img = cv_imread('./OpenCV应用项目/基于OpenCV的图像覆盖/pic.jpg')

img[50:50 + resize_img.shape[0], 50:50 + resize_img.shape[1]] = resize_img

cv2.imshow('resize', img)
cv2.waitKey(0)
