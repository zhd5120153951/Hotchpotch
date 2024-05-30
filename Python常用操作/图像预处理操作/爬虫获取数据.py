import requests
from PIL import Image
import os
from io import BytesIO

# 设置本地保存路径
local_path = 'D:\\FilePackage\\datasets\\phonecall\\train'
if not os.path.exists(local_path):
    os.makedirs(local_path)

# 计数器，记录已爬取的图片数量
count = 0

# 循环爬取图片直到达到5000张
while count < 5000:
    # 发起GET请求获取图像数据
    response = requests.get(
        'https://www.vcg.com/creative-image/dashouji/')  # 请替换为实际的图片URL

    # 检查请求是否成功
    if response.status_code == 200:
        # 使用Pillow库打开图像并检查尺寸
        img = Image.open(BytesIO(response.content))
        width, height = img.size

        # 确保图像尺寸大于640*640，否则跳过该图像
        if width > 640 and height > 640:
            # 将图像保存到本地路径
            img.save(os.path.join(local_path, f'phone_image_{count}.jpg'))
            print(f'成功保存电话图像：phone_image_{count}.jpg')
            count += 1
        else:
            print(f'不符合要求的图像，尺寸过小')
    else:
        print(f'请求失败')

print('已爬取5000张图片，程序结束。')
