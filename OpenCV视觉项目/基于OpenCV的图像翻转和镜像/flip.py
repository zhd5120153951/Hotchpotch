'''
@FileName   :test.py
@Description:
@Date       :2021/09/19 09:04:15
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageFlip(object):
    def __init__(self, img_path) -> None:
        self.img_path = img_path

    def read_img(self, gray_scale=False):
        img = cv2.imdecode(np.fromfile(self.img_path, np.uint8), -1)  #rgb格式
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if gray_scale:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    #自己实现镜像
    def mirror_lr(self, with_polt=True, gray_scale=False):
        img_rgb = self.read_img(gray_scale)
        img_mirror_lr = np.fliplr(img_rgb)
        if with_polt:
            self.plot_it(orig_matrix=img_rgb, trans_matrix=img_mirror_lr, head_text='mirror_lr')

    def mirror_ud(self, with_plot=True, gray_scale=False):
        img_rgb = self.read_img(gray_scale)
        img_mirror_ud = np.flipud(img_rgb)
        if with_plot:
            self.plot_it(orig_matrix=img_rgb, trans_matrix=img_mirror_ud, head_text='mirror_ud')

    def plot_it(self, orig_matrix, trans_matrix, head_text, gray_scale=False):
        fig = plt.figure(figsize=(10, 20))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis('off')
        ax1.title.set_text('original')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis('off')
        ax2.title.set_text(head_text)

        if not gray_scale:
            ax1.imshow(orig_matrix)
            ax2.imshow(trans_matrix)

        else:
            ax1.imshow(orig_matrix, cmap='gray')
            ax2.imshow(trans_matrix, cmap='gray')


imgFlip = ImageFlip(img_path='./OpenCV视觉项目/基于OpenCV的图像翻转和镜像/7.jpg')
imgFlip.mirror_lr()
imgFlip.mirror_lr(gray_scale=True)
imgFlip.mirror_ud()
imgFlip.mirror_ud(gray_scale=True)

# img = cv_imread('./OpenCV视觉项目/基于OpenCV的图像翻转和镜像/7.jpg')

# img = read_img(img)
# cv2.namedWindow('src', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('src', img.shape[1], img.shape[0])
# cv2.imshow('src', img)

# img_flip_lr = np.fliplr(img)  #这个方法比cv2.flip()执行要快
# cv2.namedWindow('flip_lr', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('flip_lr', img_flip_lr.shape[1], img_flip_lr.shape[0])
# cv2.imshow('flip_lr', img_flip_lr)

# cv2.waitKey(0)
