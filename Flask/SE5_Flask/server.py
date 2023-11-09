# 服务器端先运行server.py,发送端再运行client.py
from flask import request, Flask
# from flask_cors import CORS
import base64
from grpc import method_handlers_generic_handler
import numpy as np
import cv2
import logging

app = Flask(__name__)
# CORS(app, supports_credentials=True)
# 解决浏览器输出乱码问题
app.config['JSON_AS_ASCII'] = False

# logging.basicConfig(filename='./WebPost/serverApp.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s-%(levelname)s-%(message)s')


def get_data():
    if request.get_json(silent=True):
        return request.get_json(silent=True)
    else:
        if request.form:
            return request.form
        else:
            return request.args


# @app.route("/", methods=['POST', 'GET'])
# def get_frame():
#     # 解析图片数据
#     # img = base64.b64decode(str(request.form['file1']))
#     # img=str(request.form['file1'])

#     # 读取单张图像
#     # file = request.files['file']
#     # file.save('./WebPost/test.jpg')
#     # return {'sim': "0.8"}--原始代码

#     # 循环读取图像
#     # cn = 0
#     # while True:
#     #     file = request.files['file']
#     #     if file is None:
#     #         continue
#     #     cn += 1
#     #     file.save('./WebPost/box_region/' + str(cn) + '.jpg')
#     #     return {'index': cn}  #服务器反馈给子节点得字典

#     global cn
#     data = get_data()
#     try:
#         device = data.get("device", "")  # 两个参数对应key-value
#         # image_base64_str = data.get("image", "")
#         date = data.get("date", "")
#         img = data.get("img", "")

#         img_bytes = base64.b64decode(img)
#         img_np = np.frombuffer(img_bytes, dtype=np.uint8)
#         img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#         # image_bytes = base64.b64decode(image_base64_str)
#         # image_np = np.frombuffer(image_bytes, dtype=np.uint8)
#         # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

#         # print("device id = ", device)
#         logging.info("device name:{}  date:{}".format(device, date))
#         # print(image.shape)
#         # print("img_path", img_path)#太长
#         # print("ratio = ", ratio)

#         if len(img.shape) != 0:
#             # cv2.imshow("decode img", img)

#             cv2.imwrite('./WebPost/box_region/' + str(cn + 1) + '.jpg', img)
#             cn += 1
#             # 解析的图像信息
#             # print("img height = ", img.shape[0])
#             # print("img width = ", img.shape[1])
#             # cv2.waitKey(0)
#         return {'index': cn}
#     except:
#         pass


@app.route(rule="/list", methods=['GET'])
def get_data():
    data = get_data()
    print(data)


if __name__ == "__main__":
    cn = 0
    app.run("0.0.0.0", port=6666)
