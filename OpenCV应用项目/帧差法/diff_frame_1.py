'''
@FileName   :diff_frame_1.py
@Description:帧差法--差1帧
@Date       :2022/11/02 11:36:32
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import time

# 创建视频对象
# video = cv2.VideoCapture(
#     "D:\\FilePackage\\BaiduDiskDownload\\3#\\3#站控制室围墙_20230727120000-20230727235959_1.mp4")
video = cv2.VideoCapture('./gas2.mp4')

# 获取第一帧
ret, frame1 = video.read()

# 转换为灰度图像
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("Frame Diff", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame Diff", 640, 480)
cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Threshold", 640, 480)

# 循环处理视频帧
while True:
    # 读取下一帧
    ret, frame2 = video.read()
    if not ret:
        break

    # 转换为灰度图像
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算帧差
    frame_diff = cv2.absdiff(gray1, gray2)

    # 应用阈值处理
    # _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]

    # 进行形态学处理，去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    threshold = cv2.erode(threshold, kernel)

    threshold = cv2.dilate(threshold, kernel)

    # 寻找阈值图像中的轮廓并进行过滤
    contours, _ = cv2.findContours(
        threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        # 标记运动物体
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # 显示结果
        # cv2.imshow("Frame Diff", frame_diff)
        # cv2.imshow("Threshold", threshold)
        cv2.imshow("Frame Diff", frame2)
        cv2.imshow("Threshold", threshold)

        # time.sleep(0.1)
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 更新前一帧
    gray1 = gray2

# 释放视频对象和窗口资源
video.release()
cv2.destroyAllWindows()
