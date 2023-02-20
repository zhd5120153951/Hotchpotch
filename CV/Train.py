'''
@FileName   :Train.py
@Description:训练代码
@Date       :2023/02/16 21:08:35
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Model import *
from Test import predict


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个6维向量，前三个数大于后三个数--为正样本
#                    前三个数等于后三个数--为零样本
#                    前三个数小于后三个数--为负样本
def build_sample():
    a = 0
    b = 0
    c = 0
    x = np.random.random(6)
    if x[0] > x[1] and x[2] > x[3] and x[4] > x[5]:
        a += 1
        return x, 1, a  #1是正样本
    elif x[0] < x[1] and x[2] < x[3] and x[4] < x[5]:
        b += 1
        return x, -1, b  #0是负样本
    else:
        c += 1
        return x, 0, c


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y, z = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y), z


# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        # 模型预测
        y_pred = model(x)
        # 预测值与真实标签进行对比
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            elif float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 6  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型--已经有train方法了
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "TriClass.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843, 0.23454322],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681, 0.66666666],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.99871392, 0.77777777],
                [0.1349776, 0.59416669, 0.62579291, 0.41567412, 0.7358894, 0.88888888],
                [0.8349776, 0.59416669, 0.62579291, 0.41567412, 0.2358894, 0.18888888]]

    predict("TriClass.pth", test_vec)
