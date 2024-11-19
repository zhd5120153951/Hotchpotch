'''
@FileName   :2lines_intersect.py
@Description:两条线段是否相交
@Date       :2024/10/09 17:53:28
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import numpy as np

# 计算叉积


def cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

# 判断点C是否在线段AB的方向上


def direction(a, b, c):
    return cross_product((b[0] - a[0], b[1] - a[1]), (c[0] - a[0], c[1] - a[1]))

# 判断两条线段是否相交


def is_intersect(p1, p2, p3, p4):
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    # 检查两条线段是否互在对方的两侧
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    # 特殊情况：线段共线但有重叠部分
    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True

    return False

# 判断点p是否在线段ab上


def on_segment(a, b, p):
    if min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]):
        return True
    return False


# 测试
p1 = (1, 1)
p2 = (5, 5)
p3 = (2, 1)
p4 = (5, 2)

if is_intersect(p1, p2, p3, p4):
    print("两条线段相交")
else:
    print("两条线段不相交")
