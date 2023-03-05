'''
@FileName   :data_get.py
@Description:网络爬虫，获取数据集
@Date       :2023/03/04 10:07:58
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import requests
import re
import os

header = {
    'User-Agent':
    'Mozilla/5.0(Windows NT 10.0; Win64;X64) AppleWebKit/537.36(KHTML,like Gecko) Chrome/84.0.4147.125 Safari/537.36'
}
name = input("请输入要爬取的图像类别：")
num = 0
num1 = 0
num2 = 0
x = input("请输入要爬取的图像数量(1==60张图，2==120张图)：")
list1 = []
for i in range(int(x)):
    name1 = os.getcwd()  # 这个函数返回的是C盘的目录，可以把这里替换为其他目录
    name2 = os.path.join(name1, 'data/' + name)
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + name + '&pn=' + str(i * 30)
    res = requests.get(url, header)
    html1 = res.content.decode()
    a = re.findall('"objURL":"(.*?)",', html1)
    if not os.path.exists(name2):
        os.makedirs(name2)
    for b in a:
        try:
            b_1 = re.findall('https:(.*?)&', b)
            b_2 = ''.join(b_1)
            if b_2 not in list1:
                num = num + 1
                img = requests.get(b)
                f = open(os.path.join(name1, 'data/' + name, name + str(num) + '.jpg'), 'ab')
                print('---------正在下载第' + str(num) + '张图片----------')
                f.write(img.content)
                f.close()
                list1.append(b_2)
            elif b_2 in list1:
                num_1 = num_1 + 1
                continue
        except Exception as e:
            print('---------第' + str(num) + '张图片无法下载----------')
            num_2 = num_2 + 1
            continue

print('下载完成,总共下载{}张,成功下载:{}张,重复下载:{}张,下载失败:{}张'.format(num + num_1 + num_2, num, num_1, num_2))
