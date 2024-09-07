'''
@FileName   :portRTSP_Mouse.py
@Description:
@Date       :2022/09/25 14:45:30
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
import gradio as gr
import numpy as np
import json


#处理rtsp和mouse事件
def process_rtsp_mouse(rtsp_url, output):
    #创建VideoCapture对象
    cap = cv2.VideoCapture(rtsp_url)

    #定义鼠标回调函数--尝试在外部定义
    def mouse_callback(event, x, y, flag, param):
        global points, Polygon_point, count
        if event == cv2.EVENT_LBUTTONDBLCLK:
            polygon = zip(Polygon_point, points)
            with open("./Gradio/" + output, 'w') as f:
                json.dump(dict(polygon), f)
            # with open("./Gradio/" + output, 'a') as f:
            #     f.write(f"{x},{y}\n")
        elif event == cv2.EVENT_LBUTTONDOWN:
            count += 1
            points.append((x, y))
            pt = 'pt' + str(count)
            Polygon_point.append(pt)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points = []
            Polygon_point = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        temp_frame = frame.copy()
        if len(points) > 0:
            cv2.polylines(temp_frame, np.array([points]), isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.namedWindow("rtsp")
        cv2.resizeWindow("rtsp", 1920, 1080)
        cv2.imshow("rtsp", temp_frame)
        cv2.setMouseCallback("rtsp", mouse_callback)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


iface = gr.Interface(fn=process_rtsp_mouse, inputs=["text", "text"], outputs=None, title="rtsp stream")

if __name__ == "__main__":
    points = []
    Polygon_point = []
    count = 0
    output = None
    iface.launch()