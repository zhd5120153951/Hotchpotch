import gradio as gr
import logging
import numpy as np
import cv2
# logging.basicConfig函数各参数:
# filename: 指定日志文件名
# filemode: 和file函数意义相同，指定日志文件的打开模式，'w'或'a'
# format: 指定输出的格式和内容，format可以输出很多有用信息，如上例所示:
#  %(levelno)s: 打印日志级别的数值
#  %(levelname)s: 打印日志级别名称
#  %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
#  %(filename)s: 打印当前执行程序名
#  %(funcName)s: 打印日志的当前函数
#  %(lineno)d: 打印日志的当前行号
#  %(asctime)s: 打印日志的时间
#  %(thread)d: 打印线程ID
#  %(threadName)s: 打印线程名称
#  %(process)d: 打印进程ID
#  %(message)s: 打印日志信息
# datefmt: 指定时间格式，同time.strftime()
# level: 设置日志级别，默认为logging.WARNING
# stream: 指定将日志的输出流，可以指定输出到sys.stderr,sys.stdout或者文件，默认输出到sys.stderr，当stream和filename同时指定时，stream被忽略

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(thread)d %(threadName)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename='myapp.log',
                    filemode='w')  # a是追加,w是覆盖
# 创建一个日志记录器
logger = logging.getLogger('my_logger')


def greet(name, is_moring, temperatue):
    salutation = "Good morning" if is_moring else "Good evening"

    greeting = f"{salutation} {name}. It is {temperatue} degrees today"
    celsius = (temperatue - 32) * 5 / 9
    return greeting, round(celsius, 2)


# gr.Interface(fn=greet, inputs="text", outputs="text")--单个输入输出框
# gr.Textbox()=="text"带个格式的text


def img_proc(input_img):
    sepia_filter = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


# streaming
def stream_proc():
    cap = cv2.VideoCapture("rtsp://admin:great123@192.168.8.201")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        return np.flipud(frame)


demo = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(lines=2, placeholder="please input here...", label="your input"), "checkbox",
            gr.Slider(0, 100)],
    outputs=["text", "number"],
)

demo2 = gr.Interface(fn=img_proc, inputs=gr.Image(
    shape=(640, 640)), outputs="image")
# rtsp://admin:great123@192.168.8.201
demo3 = gr.Interface(fn=stream_proc, inputs=gr.Image(
    shape=(640, 640)), outputs="image")
if __name__ == "__main__":
    # app, local_url, share_url = demo.launch()
    # logging.info(app)
    # logging.info(share_url)
    # logging.info(local_url)

    demo2.launch()
    # demo3.launch()
