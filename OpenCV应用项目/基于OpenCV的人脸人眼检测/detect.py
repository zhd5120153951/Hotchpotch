'''
@FileName   :detect.py
@Description:
@Date       :2022/10/11 17:35:49
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
'''
目前设计的退出程序的方式为双击图像退出和按下esc退出，程序不能使用单击关闭按钮的方式退出

当没有识别到人脸或识别到两张及两张以上的人脸时将出发报警，左右眼同理

'''

# ---------------- 导入模块 ----------------

# import RPi.GPIO as GPIO--在电脑端测试
import multiprocessing
import cv2
import time
import sys

# ---------------- 全局变量、摄像头、GPIO初始化等 ----------------

# 初始化摄像头并设置相关参数
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 用于人脸、眼睛识别的哈尔特征级联分类器(Harr Cascade)
# opencv中带有多个分类器，效果各有不同，请根据实际情况选取

# haarcascade_face_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt.xml"
# haarcascade_eyeL_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_lefteye_2splits.xml"
# haarcascade_eyeR_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_righteye_2splits.xml"

haarcascade_face_path = "D:\\opencv3.4.16\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"
haarcascade_eyeL_path = "D:\\opencv3.4.16\\opencv\\sources\\data\\haarcascades\\haarcascade_lefteye_2splits.xml"
haarcascade_eyeR_path = "D:\\opencv3.4.16\\opencv\\sources\\data\\haarcascades\\haarcascade_righteye_2splits.xml"

# haarcascade_face_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt.xml"
# haarcascade_face_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
# haarcascade_face_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt2.xml"
# haarcascade_face_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml"
# haarcascade_eyeL_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_lefteye_2splits.xml"
# haarcascade_eyeR_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_righteye_2splits.xml"
# haarcascade_eyeL_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_mcs_lefteye.xml"
# haarcascade_eyeR_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_mcs_righteye.xml"
# haarcascade_eye_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_eye.xml"
# haarcascade_eye_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
# haarcascade_eye_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_mcs_eyepair_small.xml"
# haarcascade_eye_path = "/home/pi/opencv/opencv-3.4.0/data/haarcascades/haarcascade_mcs_eyepair_big.xml"

# 初始化opencv的级联分类器
face_cascade = cv2.CascadeClassifier(haarcascade_face_path)
eyeL_cascade = cv2.CascadeClassifier(haarcascade_eyeL_path)
eyeR_cascade = cv2.CascadeClassifier(haarcascade_eyeR_path)

# 全局变量，用于保存识别出的人脸、左右眼的坐标
face_region = [0, 0, 0, 0]
eyeL_region = [0, 0, 0, 0]
eyeR_region = [0, 0, 0, 0]

# 全局变量，用于保存检测结果
DetectResult = 0
# 检测结果定义
FIND_NONE = 0x00  # 无结果
FIND_FACE = 0x01  # 脸
FIND_EYEL = 0x02  # 左眼
FIND_EYER = 0x04  # 右眼

# 全局变量，报警检测的时间，单位：秒
TIMER_THRESHOLD = 5.0

# 初始化IO口

# # 关闭警告
# GPIO.setwarnings(False)
# # BOARD编号方式，基于插座引脚编号
# GPIO.setmode(GPIO.BOARD)
# # IO口设置为输出模式
# GPIO.setup(13, GPIO.OUT)  # GPIO13为LED
# GPIO.setup(15, GPIO.OUT)  # GPIO15为蜂鸣器
# # 输出低电平
# GPIO.output(13, GPIO.LOW)
# GPIO.output(15, GPIO.LOW)

# ---------------- 类定义 ----------------


# 自己写的定时器类
class timer():
    # 构造函数
    def __init__(self):
        self.status = 0
        self.StartTime = 0
        self.StopTime = 0

    # 开始计时
    def start(self):
        if (self.status == 0):
            self.StartTime = time.time()
            self.status = 1

    # 结束计时
    def stop(self):
        if (self.status == 1):
            self.StopTime = time.time()
            self.status = 0
        return self.StopTime - self.StartTime

    # 定时器启动时获取已经计时的时间，定时器停止时获取从开始到停止的时间
    def getTimer(self):
        if (self.status == 0):
            return self.StopTime - self.StartTime
        else:
            return time.time() - self.StartTime

    # 重置定时器
    def reset(self):
        self.status = 0
        self.StartTime = 0
        self.StopTime = 0


# ---------------- 函数定义 ----------------


# 报警
# event: 信号量，保证GPIO不会被并发调用
def alarm(event):
    if (event.is_set()):
        return
    else:
        event.set()  # 获取信号量
        for i in range(0, 4):
            # 让蜂鸣器和LED按一定规律报警
            # GPIO.output(13, GPIO.HIGH)
            # GPIO.output(15, GPIO.LOW)
            # time.sleep(0.2)
            # GPIO.output(13, GPIO.LOW)
            # GPIO.output(15, GPIO.HIGH)
            print('报警...')
            time.sleep(0.2)

        # 关闭蜂鸣器和LED
        # GPIO.output(13, GPIO.LOW)
        # GPIO.output(15, GPIO.LOW)

        event.clear()  # 释放信号量


# 人脸识别
# img: 待识别的图片
def GetFace(img):
    # 引用全局变量
    global face_region, DetectResult

    # 参数image:要检测的图片，一般为灰度图
    # 参数scaleFactor:缩放因子，对图片进行缩放，默认为1.1
    # 参数minNeighbors:最小邻居数，默认为3
    # 参数flags:兼容老版本的一个参数，在3.0以后的版本中没用，默认为0
    # 参数minSize:检测的最小尺寸
    # 参数maxSize:检测的最大尺寸
    faces = face_cascade.detectMultiScale(
        image=img
        # scaleFactor = 1.15,
        # minNeighbors = 3,
        # flags = 0,
        # minSize = (20,20),
        # maxSize = (200,200)
    )

    if (len(faces) != 1):
        # 没有识别到脸或识别到多张脸则清除标志位，并将脸部坐标清零
        DetectResult &= ~FIND_FACE
        face_region = [0, 0, 0, 0]
    else:
        # 否则设置相关标志位，保存脸部坐标
        DetectResult |= FIND_FACE
        face_region = faces[0]


# 眼睛识别
# img: 待识别的图片
def GetEyes(img):
    # 引用全局变量
    global eyeL_region, eyeR_region, DetectResult

    eyeL = eyeL_cascade.detectMultiScale(
        image=img
        # scaleFactor = 1.15,
        # minNeighbors = 3,
        # flags = 0,
        # minSize = (20,20),
        # maxSize = (200,200)
    )

    eyeR = eyeR_cascade.detectMultiScale(
        image=img
        # scaleFactor = 1.15,
        # minNeighbors = 3,
        # flags = 0,
        # minSize = (20,20),
        # maxSize = (200,200)
    )

    # 处理左眼识别结果
    if (len(eyeL) != 1):
        # 未识别出左眼或识别出多个左眼则清除标志位，并将左眼坐标清零
        DetectResult &= ~FIND_EYEL
        eyeL_region = [0, 0, 0, 0]
    else:
        # 设置标志位，保存眼部坐标，计算眼睛相对于脸部的坐标
        DetectResult |= FIND_EYEL
        eyeL_region = eyeL[0]
        eyeL_region[0] = eyeL_region[0] + face_region[0]
        eyeL_region[1] = eyeL_region[1] + face_region[1] + int(face_region[3] * 0.2)

    # 处理右眼识别结果
    if (len(eyeR) != 1):
        # 未识别出右眼或识别出多个右眼则清除标志位，并将右眼坐标清零
        DetectResult &= ~FIND_EYER
        eyeR_region = [0, 0, 0, 0]
    else:
        # 设置标志位，保存眼部坐标，计算眼睛相对于脸部的坐标
        DetectResult |= FIND_EYER
        eyeR_region = eyeR[0]
        eyeR_region[0] = eyeR_region[0] + face_region[0]
        eyeR_region[1] = eyeR_region[1] + face_region[1] + int(face_region[3] * 0.2)


# 提取脸部图像
# img: 待提取的图片
# face_region_img: 提取出的脸部图片
def SelectFace(img):
    global face_region
    face_region_img = img[face_region[1]:face_region[1] + face_region[3],
                          face_region[0]:face_region[0] + face_region[2]]
    return face_region_img


# 从识别出的脸上选取眼睛的区域图像
# img: 待提取的图片
# eyes_region_img: 提取出的眼睛图片
def SelectEyes(img):
    # 设人脸图像的长宽均为1个单位长度，x向右为正方向，y向下为正方向
    # 眼睛的区域一般位于人脸的y=0.2至y=0.6的范围内

    # 获取图像的长宽
    y, x = img.shape

    # 计算要选取的范围
    y1 = int(y * 0.2)
    y2 = int(y * 0.6)

    eyes_region_img = img[y1:y2, 0:x]  # 提取区域img[y1:y2,x1:x2]

    return eyes_region_img


# 绘制人脸，眼睛的边框
def DrawBoundary(img, face, eye1, eye2):
    cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 3)
    cv2.rectangle(img, (eye1[0], eye1[1]), (eye1[0] + eye1[2], eye1[1] + eye1[3]), (255, 0, 0), 3)
    cv2.rectangle(img, (eye2[0], eye2[1]), (eye2[0] + eye2[2], eye2[1] + eye2[3]), (255, 0, 0), 3)
    return img


# 鼠标事件处理函数
def OnMouseAction(event, x, y, flags, param):
    # 双击鼠标退出程序
    if (event == cv2.EVENT_LBUTTONDBLCLK):
        # 释放摄像头
        cap.release()
        # 销毁所有窗口
        cv2.destroyAllWindows()
        # 关闭蜂鸣器、LED，GPIO清零
        # GPIO.output(13, GPIO.LOW)
        # GPIO.output(15, GPIO.LOW)
        # GPIO.cleanup()

        # 退出
        sys.exit()


# 主函数
def main():
    # 引用全局变量
    global face_region, eyeL_region, eyeR_region, DetectResult

    # 设置窗口的名字
    cv2.namedWindow("detection")

    # 绑定鼠标事件
    cv2.setMouseCallback("detection", OnMouseAction)

    # 初始化计时器
    face_timer = timer()
    eyes_timer = timer()

    # 信号量
    AlarmEvent = multiprocessing.Event()

    # 程序主循环
    while (True):
        # 读取摄像头的图像，这个函数有两个返回参数，一个是检测结果，一个是图像
        ret, frame = cap.read()
        # 翻转图像，1:水平翻转，0:垂直翻转，-1:水平垂直翻转
        # frame = cv2.flip(frame, 0)

        # 转为灰色减小计算压力
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 识别脸部图像
        GetFace(gray)

        if (DetectResult & FIND_FACE == FIND_FACE):
            # 如果检测到人脸，进一步提取眼部区域的图像并检测眼睛
            face_img = SelectFace(gray)
            eye_img = SelectEyes(face_img)
            GetEyes(eye_img)
        else:
            # 否则清除眼睛的检测结果
            # 如果这里不清除，会导致上一次眼睛检测的结果叠加在没有识别出人脸的图像上
            DetectResult &= ~FIND_EYEL
            DetectResult &= ~FIND_EYER
            eyeL_region = [0, 0, 0, 0]
            eyeR_region = [0, 0, 0, 0]

        # 计时与报警的相关代码
        if (DetectResult & FIND_FACE == FIND_NONE):
            # 未检测到人脸则启动计时器
            face_timer.start()
        else:
            # 否则重置计时器
            face_timer.reset()

        if (DetectResult & (FIND_EYEL | FIND_EYER) == FIND_NONE):
            # 未检测到左右眼则启动计时器
            eyes_timer.start()
        else:
            # 否则重置计时器
            eyes_timer.reset()

        if (face_timer.getTimer() > TIMER_THRESHOLD):
            # 如果5秒之内没检测到人脸则报警
            # 保证人脸检测能继续运行，这里启动一个新的进程执行报警程序
            p = multiprocessing.Process(target=alarm, args=(AlarmEvent, ))
            p.start()
            p.join(0)
        elif (eyes_timer.getTimer() > TIMER_THRESHOLD):
            # 如果5秒之内没检测到眼睛则报警
            # 保证人脸检测能继续运行，这里启动一个新的进程执行报警程序
            p = multiprocessing.Process(target=alarm, args=(AlarmEvent, ))
            p.start()
            p.join(0)
        else:
            pass

        # 画出脸、眼睛的范围并显示
        marked_face = DrawBoundary(frame, face_region, eyeL_region, eyeR_region)

        # cv2.imshow("detection", frame) # 显示摄像头获取的原始图像
        # cv2.imshow("detection", eye_img) # 显示眼部区域图像
        cv2.imshow("detection", marked_face)  # 显示标记出的脸部、眼睛图像

        # 按esc键退出程序
        if (cv2.waitKey(1) & 0xFF == 27):
            break

    # 释放摄像头
    cap.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()
    # 关闭蜂鸣器、LED，GPIO清零
    # GPIO.output(13, GPIO.LOW)
    # GPIO.output(15, GPIO.LOW)
    # GPIO.cleanup()
    # 退出
    sys.exit()


# ---------------- 程序入口 ----------------
if (__name__ == "__main__"):
    main()
