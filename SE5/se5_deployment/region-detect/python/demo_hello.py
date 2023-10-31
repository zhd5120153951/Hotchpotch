import cv2
import numpy as np
import gradio as gr
import json
#暂时无法实现通过端口访问盒子的rtsp流,并实现区域绘制
# 1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_aarch64
# 2. Rename the downloaded file to: frpc_linux_aarch64_v0.2
# 3. Move the file to this location: /home/linaro/.local/lib/python3.8/site-packages/gradio

#text
def greet(name):
    return "hello "+name+" gradio"
#video
def read_video_stream():
    url = "rtsp://admin:jiankong123@192.168.23.10:554/Streaming/Channels/101"
    cap = cv2.VideoCapture(url)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        # 在 Gradio 应用中显示视频帧
        yield frame[:,:,:-1]
    cap.release()

#处理rtsp和mouse事件
def process_rtsp_mouse(rtsp_url,output):
    cap = cv2.VideoCapture(rtsp_url)

    def mouse_callback(event,x,y,flag,param):
        global points,Polygon_point,count
        if event == cv2.EVENT_LBUTTONDBLCLK:
            polygon = zip(Polygon_point,points)
            with open("../region/"+output,'w') as f:
                json.dump(dict(polygon),f)
        elif event == cv2.EVENT_LBUTTONDOWN:
            count += 1
            points.append((x,y))
            pt = 'pt'+str(count)
            Polygon_point.append(pt)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points = []
            Polygon_point = []

    while True:
        ret,frame = cap.read()
        if not ret:
            break
        temp_frame = frame.copy()
        if len(points)>0:
            cv2.polylines(temp_frame,np.array([points]),isClosed=True,color=(0,255,0),thickness=1)
        cv2.namedWindow("rtsp")
        cv2.resizeWindow("rtsp",1920,1080)
        cv2.imshow("rtsp",temp_frame)
        cv2.setMouseCallback("rtsp",mouse_callback)
        if cv2.waitKey(0) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# demo = gr.Interface(fn=greet,inputs="text",outputs="text")
# demo = gr.Interface(fn=read_video_stream,inputs=None,outputs="video")
demo2 = gr.Interface(fn=process_rtsp_mouse,inputs=["text","text"],outputs=None,title="rtsp stream")

if __name__ == "__main__":
    # demo.launch()
    points= []
    Polygon_point = []
    count = 0
    output = None
    # demo.queue().launch()
    demo2.launch(share=True)