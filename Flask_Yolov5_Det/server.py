'''
@FileName   :server.py
@Description:
@Date       :2022/10/18 13:19:02
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
#编写server.py文件，封装服务端程序
from flask import *
import cv2
import logging as rel_log
from datetime import timedelta
from flask_cors import CORS
from detect import VideoCamera

import base64
import time
import numpy as np
import json

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def get_frame():
    # 解析图片数据
    # img = base64.b64decode(str(request.form['file1']))
    # img=str(request.form['file1'])
    file = request.files['file']
    file.save('result/test.jpg')
    return {'sim': "0.8"}


if __name__ == "__main__":
    print(("* Loading yolov5 model and Flask starting server..." "please wait until server has fully started"))
    app.run("0.0.0.0", port=6666)
