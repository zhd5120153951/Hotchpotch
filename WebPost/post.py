import requests
import json
import base64


def img2base64(img_path):
    # 读取图片文件
    with open(img_path, 'rb') as image_file:
        # 将图片内容进行base64编码
        encoded_image = base64.b64encode(image_file.read())

        return encoded_image.decode("utf-8")  #把byte转换为字符串


if __name__ == '__main__':
    encoded_string = img2base64("./1.jpg")
    # 打印编码后的图片内容
    # print(encoded_string)
    data = {'device': 'Dell-G15', 'ratio': 1.2, 'img_path': encoded_string}
    header = {"Content-Type": "application/json; charset=utf-8"}
    # print(json.dumps(data))
    try:
        requests.post("http://127.0.0.1:5002/flask", json.dumps(data), headers=header)
    except Exception as e:
        print(e)
