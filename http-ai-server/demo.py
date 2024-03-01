import time
from flask import Flask, request
from flask_cors import CORS
import cv2
import threading

app = Flask(__name__)
CORS(app)

# 默认参数
rtspUrl = ""
thresh = 0.5
detInterval = 1
# 视频处理函数--最好的做法是对流做轮询推流检测


def proc_video(rtspUrl, thresh, detInterval):
    video = cv2.VideoCapture(rtspUrl)
    detector = "Ai model"

    while True:
        ret, frame = video.read()  # py可以仿照yolo官方的拉流写法
        if ret:
            # 处理视频帧
            # 调用检测模型对每一帧进行目标检测
            detections = detector.detect(frame)

            for detection in detections:
                if detection.confidence > thresh:
                    # 触发报警事件
                    trigger_alarm()
        # 等待间隔
        time.sleep(detInterval)


def trigger_alarm():
    print("报警产生...")


@app.route('/config', methods=["POST"])
def config():
    global rtspUrl, thresh, detInterval
    if 'rtspUrl' in request.json:
        rtspUrl = request.json['rtspUrl']
        print(rtspUrl)
    if 'thresh' in request.json:
        thresh = request.json['thresh']
        print(thresh)
    if 'detInterval' in request.json:
        detInterval = request.json['detInterval']
        print(detInterval)
    if 'video_thread' in globals():
        video_thread.join()
    video_thread = threading.Thread(
        target=proc_video, args=(rtspUrl, thresh, detInterval))
    video_thread.start()

    return {'message': 'Configuration updated'}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
