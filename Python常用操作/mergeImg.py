'''
@FileName   :mergeImg.py
@Description:合并小图成大图
@Date       :2023/10/26 13:37:03
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
import numpy as np

# 读取多个小尺寸图像
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')
image3 = cv2.imread('image3.jpg')

# 确保所有小图像具有相同的尺寸，如果不同，可以使用cv2.resize()调整它们

# 获取每个小图像的宽度和高度
width, height = image1.shape[1], image1.shape[0]

# 计算大图像的尺寸
new_width = width * 3
new_height = height

# 创建一个空的大图像
large_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

# 将小图像复制到大图像中
large_image[0:height, 0:width] = image1
large_image[0:height, width:2 * width] = image2
large_image[0:height, 2 * width:3 * width] = image3

# 保存合并后的大图像
cv2.imwrite('large_image.jpg', large_image)

# 显示大图像（可选）
cv2.imshow('Large Image', large_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
