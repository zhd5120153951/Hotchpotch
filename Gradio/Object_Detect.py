import gradio as gr
import pandas as pd
from skimage import data
from ultralytics.yolo.data import utils

model = YOLO('yolov8n.pt')

#load class_names
yaml_path = str(Path(ultralytics.__file__).parent / 'datasets/coco128.yaml')
class_names = utils.yaml_load(yaml_path)['names']


def detect(img):
    if isinstance(img, str):
        img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
    result = model.predict(source=img)
    if len(result[0].boxes.boxes) > 0:
        vis = plots.plot_detection(img, boxes=result[0].boxes.boxes, class_names=class_names, min_score=0.2)
    else:
        vis = img
    return vis


with gr.Blocks() as demo:
    gr.Markdown("# yolov8目标检测演示")

    with gr.Tab("捕捉摄像头喔"):
        in_img = gr.Image(source='webcam', type='pil')
        button = gr.Button("执行检测", variant="primary")

        gr.Markdown("## 预测输出")
        out_img = gr.Image(type='pil')

        button.click(detect, inputs=in_img, outputs=out_img)

gr.close_all()
demo.queue(concurrency_count=5)
demo.launch()
