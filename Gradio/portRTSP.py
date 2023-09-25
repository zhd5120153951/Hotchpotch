'''
@FileName   :portRTSP.py
@Description:
@Date       :2022/09/25 14:45:53
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import gradio as gr


def read_rtsp(rtsp_url):
    # 打开RTSP视频流
    cap = cv2.VideoCapture("rtsp://admin:jiankong123@192.168.23.15:554/Streaming/Channels/101")

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            break

        # 将帧转换为图像
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 显示图像
        cv2.imshow("RTSP Video", img)

        # 按下q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


iface = gr.Interface(fn=read_rtsp, inputs="text", outputs=None)
iface.launch()
