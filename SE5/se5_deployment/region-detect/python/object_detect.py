# flag后--输入和输出保存flag/log.csv中


# import gradio as gr
# #输入文本处理程序
# def greet(name):
#     return "Hello " + name + "!"
# #接口创建函数
# #fn设置处理函数，inputs设置输入接口组件，outputs设置输出接口组件
# #fn,inputs,outputs都是必填函数
# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()


# import gradio as gr

# def greet(name):
#     return "Hello " + name + "!"

# iface = gr.Interface(
#     fn=greet,
#     inputs=gr.inputs.Textbox(lines=2, placeholder="Name Here..."),
#     outputs="text",
# )
# if __name__ == "__main__":
#     app, local_url, share_url = iface.launch()
#     print(app)
#     print(local_url)
#     print(share_url)


# import gradio as gr
# import numpy as np
# def flip(im):
#     return np.flipud(im)
# demo = gr.Interface(flip,
#     gr.Image(source="webcam", streaming=True),
#     "image",
#     live=True
# )
# demo.launch()


# import gradio as gr 
# import pandas as pd 
# from skimage import data
# from ultralytics.yolo.data import utils 
 
# model = YOLO('yolov8n.pt')
 
# #load class_names
# yaml_path = str(Path(ultralytics.__file__).parent/'datasets/coco128.yaml') 
# class_names = utils.yaml_load(yaml_path)['names']

# def detect(img):
#     if isinstance(img,str):
#         img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
#     result = model.predict(source=img)
#     if len(result[0].boxes.boxes)>0:
#         vis = plots.plot_detection(img,boxes=result[0].boxes.boxes,
#                      class_names=class_names, min_score=0.2)
#     else:
#         vis = img
#     return vis
    
# with gr.Blocks() as demo:
#     gr.Markdown("# yolov8目标检测演示")
 
#     with gr.Tab("捕捉摄像头喔"):
#         in_img = gr.Image(source='webcam',type='pil')
#         button = gr.Button("执行检测",variant="primary")
 
#         gr.Markdown("## 预测输出")
#         out_img = gr.Image(type='pil')
 
#         button.click(detect,
#                      inputs=in_img, 
#                      outputs=out_img)
        
    
# gr.close_all() 
# demo.queue(concurrency_count=5)
# demo.launch(share=True)

import gradio as gr
import cv2

def to_black(image):
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("保存web页面的配置到本地json")
    return output

interface = gr.Interface(fn=to_black, inputs=["text","text"], outputs="text")
interface.launch()




