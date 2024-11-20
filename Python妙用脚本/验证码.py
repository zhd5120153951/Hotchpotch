'''
@FileName   :验证码.py
@Description:
@Date       :2024/11/20 11:24:38
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import random
from PIL import Image, ImageDraw, ImageFont


def create_captcha_content():
    """
    生成验证码内容的函数
    :return:
    """
    CAPTCHA_text = ""
    for i in range(1, 5):  # 4位验证码
        CAPTCHA_text += chr(random.randint(0x4E00, 0x9FFF)
                            )  # 随机取中文汉字的Unicode码chr转为汉字

    return CAPTCHA_text


if __name__ == "__main__":
    # 创建新图像
    image = Image.new("RGB", (300, 100), (random.randint(0, 255),
                                          random.randint(0, 255), random.randint(0, 255)))
    draw = ImageDraw.Draw(image)  # 创建画布
    # 加载字体 是Windows自带的
    font = ImageFont.truetype(r'C:\Windows\Fonts\simhei.ttf', size=30)

    test = create_captcha_content()

    for i in test:  # 随机验证码
        # 为每个验证码字符设置不同的RGB颜色
        R = str(random.randint(0, 255))
        G = str(random.randint(0, 255))
        B = str(random.randint(0, 255))

        draw.text((random.randint(50, 250), random.randint(15, 80)),  # 摆放位置
                  text=i,
                  font=font,
                  fill="rgb(" + R + "," + G + "," + B + ")",
                  direction=None)

    # 添加干扰线条
    for i in range(1, random.randint(7, 15)):  # 线条数量在7-15间
        x, y = random.randint(0, 300), random.randint(0, 100)  # 线条起点
        x2, y2 = random.randint(0, 300), random.randint(0, 100)  # 线条终点

        # 随机颜色
        R = str(random.randint(0, 255))
        G = str(random.randint(0, 255))
        B = str(random.randint(0, 255))
        # 绘制线条 宽度为2
        draw.line((x, y, x2, y2), fill="rgb(" + R +
                  "," + G + "," + B + ")", width=2)

    print(test)
    image.show()
    image.save("CAPTCHA.jpg")
