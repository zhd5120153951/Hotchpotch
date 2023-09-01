import itertools
import threading
import time

# dic = []  # 字典池--列表


def d(count):
    '''
    字典生成存入字典池
    :param count:
    :return:
    '''
    for num in range(3, int(count) + 1):
        # word = "ABCDEFGHIJKLMNOPQRSTUVWXYZzbcdefghijklmnopqrstuvwxyz`1234567890-=~!@#$%^&*()_+,./<>?[]\{}|;':\""  # 字典生成从这里取字符
        word = "ABCDEFGHLMNOPRSTXYZabcdefghlmnoprstxyz1234567890"
        a = itertools.product(word, repeat=num)
        dic = open('./dict.txt', 'a')
        print(12)
        for i in a:
            dic.write(''.join(i) + '\n')
            # dic.append(''.join(i))
        dic.close()


def a():
    '''
    弹出字典池第0个位置
    :return:
    '''
    for i in dic:
        print(dic.pop(0))


def b():
    '''
    弹出字典池第0个位置
    :return:
    '''
    for i in dic:
        print(dic.pop(0))


def c():
    '''
    弹出字典池第0个位置
    :return:
    '''
    for i in dic:
        print(dic.pop(0))


if __name__ == '__main__':
    count = input('字典最大位数：\n')
    d = threading.Thread(target=d, name='d', args=(count, ))  # 创建线程
    # a = threading.Thread(target=a, name='a', args=())
    # b = threading.Thread(target=b, name='b', args=())
    # c = threading.Thread(target=c, name='c', args=())
    d.start()  # 开始线程
    # a.start()
    # b.start()
    # c.start()
    # while True:
    #     '''
    #     这里用来查看线程存活状态
    #     '''
    #     time.sleep(10)
    #     if a.is_alive() or b.is_alive() or c.is_alive() or d.is_alive():
    #         print('**********************************************', 'A:', a.is_alive(),
    #               '**********************************************')
    #         print('**********************************************', 'B:', b.is_alive(),
    #               '**********************************************')
    #         print('**********************************************', 'C:', c.is_alive(),
    #               '**********************************************')
    #         print('**********************************************', 'D:', d.is_alive(),
    #               '**********************************************')