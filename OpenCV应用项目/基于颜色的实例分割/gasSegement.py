import cv2
import numpy as np

# 读取图像
image = cv2.imread('./4.png')

# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值处理来分割白色烟雾区域
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# 计算白色区域像素数和图像总像素数
# white_pixels = cv2.countNonZero(binary)
# total_pixels = binary.shape[0] * binary.shape[1]

# 找到白色区域的轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#计算连通区域
for contour in contours:
    area = cv2.contourArea(contour)
    if area >= int(binary.shape[0] * binary.shape[1] * 0.03):
        #认为是gas

        # 创建一个新的图像，用白色轮廓绘制烟雾区域
        smoke_mask = np.zeros_like(image)
        cv2.drawContours(smoke_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # 从原始图像中提取烟雾区域
        smoke_area = cv2.bitwise_and(image, smoke_mask)

        # 显示原始图像和提取的烟雾区域
        cv2.imshow('Original Image', image)
        cv2.imshow('Smoke Area', smoke_area)
cv2.waitKey(0)
cv2.destroyAllWindows()
