from tkinter import NO
import cv2
import gradio as gr
import numpy as np
from sympy import true


#处理rtsp和mouse事件
def process_rtsp_mouse(rtsp_url, output):
    #创建VideoCapture对象
    cap = cv2.VideoCapture(rtsp_url)

    #定义鼠标回调函数--尝试在外部定义
    def mouse_callback(event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with open("./Gradio/" + output, 'a') as file:
                file.write(f"{x},{y}\n")

    cv2.namedWindow("rtsp")
    cv2.resizeWindow("rtsp", 1920, 1080)
    cv2.setMouseCallback("rtsp", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("rtsp", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


iface = gr.Interface(fn=process_rtsp_mouse, inputs=["text", "text"], outputs=None, title="rtsp stream")

if __name__ == "__main__":
    iface.launch()