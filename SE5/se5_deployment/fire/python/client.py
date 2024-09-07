import requests

file_path1 = './python/27.jpg'  # 图片路径
img = open(file_path1, 'rb')
res = {"file": img}
# 访问服务
res = requests.post("http://192.168.22.4:6666", files=res)  # 如 http://152.111.111.11:6666
print(res.text)

