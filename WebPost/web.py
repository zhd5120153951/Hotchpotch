# 导入对应包
from flask import Flask, request
from flask_cors import CORS
import json
import cv2
import base64
import numpy as np

# flask格式
app = Flask(__name__)

CORS(app, supports_credentials=True)
# 解决浏览器输出乱码问题
app.config['JSON_AS_ASCII'] = False
# 满足get和post请求


def form_or_json():
    if request.get_json(silent=True):
        return request.get_json(silent=True)
    else:
        if request.form:
            return request.form
        else:
            return request.args


# 代码区域
@app.route("/flask", methods=["GET", "POST"])
def new_flask():
    # 接收请求数据★★★

    data = form_or_json()
    try:
        device = data.get("device", "")
        # image_base64_str = data.get("image", "")
        ratio = data.get("ratio", float)
        img_path = data.get("img_path", "")

        img_bytes = base64.b64decode(img_path)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # image_bytes = base64.b64decode(image_base64_str)
        # image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        print("device id = ", device)
        # print(image.shape)
        # print("img_path", img_path)#太长
        print("ratio = ", ratio)

        if len(img.shape) != 0:
            cv2.imshow("decode img", img)
            #解析的图像信息
            print("img height = ", img.shape[0])
            print("img width = ", img.shape[1])
            cv2.waitKey(0)

    except:
        pass


# 一定要返回值，且请求格式为 list 格式，不然请求端格式读取不出来
# return （list）
    return []

if __name__ == '__main__':
    app.run("0.0.0.0", "5002", debug=True)