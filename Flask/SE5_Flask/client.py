import requests


if __name__ == '__main__':

    # LABELS_URL = 'http://8.137.53.211//dev-api/system/student'

    # classes = {
    #     (key, value) for (key, value)
    #     in requests.get(LABELS_URL).json().items()
    # }

    # print(classes)

    #  替换为您想要获取数据的远程服务器URL
    url = "http://8.137.53.211/prod-api/system/product/123"
    # data = dict()
    #  发送GET请求
    response = requests.get(url)
    # print(data)
    #  检查请求是否成功
    if response.status_code == 200:
        #  获取服务器返回的数据
        data = response.json()
        print("获取到的数据：",  data)
    else:
        print("请求失败，状态码：",  response.status_code)

    # file_path1 = './1.jpg'  # 图片路径
    # img = open(file_path1, 'rb')
    # res = {"file": img}
    # data = {'key1': 'value1', 'key2': 'value2'}
    # # 访问服务
    # # 如 http://152.111.111.11:6666
    # # res = requests.post("http://8.137.53.211//dev-api/system/role", files=res)
    # ret = requests.post("http://8.137.53.211//dev-api/system/role", data=res)

    # print(ret.text)
    # if ret.status_code == 200:
    #     print('数据发送成功')
    # else:
    #     print('数据发送失败')
