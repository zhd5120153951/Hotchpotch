import cv2
import numpy as np

#创建窗口
cv2.namedWindow("trackbar", cv2.WINDOW_NORMAL)
cv2.resizeWindow("trackbar", 640, 480)


#回调
def callback(value):
    print(value)


#创建三个trackbar
R = cv2.createTrackbar("R", "trackbar", 0, 255, callback)
G = cv2.createTrackbar("G", "trackbar", 0, 255, callback)
B = cv2.createTrackbar("B", "trackbar", 0, 255, callback)

#生成一张图
img = np.zeros((480, 640, 3), np.uint8)
while True:
    r = cv2.getTrackbarPos("R", "trackbar")
    g = cv2.getTrackbarPos("G", "trackbar")
    b = cv2.getTrackbarPos("B", "trackbar")

    img[:] = [b, g, r]  #这句话时把三个rgb值给img图片
    cv2.imshow("trackbar", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
#这个可已转换为HSV格式的