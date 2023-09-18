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

img = cv2.imread('E:\\Source\\Github\\test\OpenCV视觉项目\\基于OpenCV的图像覆盖\\15.jpg')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img)

img[50:150, 50:150] = [36, 72, 108]
cv2.imshow(img)

resize_img = cv2.resize(img, dsize=(100, 100))

img[50:150, 50:150] = resize_img

cv2.imshow(img)