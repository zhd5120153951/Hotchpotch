'''
@FileName   :logicregrass.py
@Description:
@Date       :2023/08/30 15:46:46
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 导入标准化方法
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归方法
from sklearn.metrics import classification_report


def load_data(filepath, names):
    # breast存放癌症数据，不默认将第一行作为列索引名，自定义列索引名
    return pd.read_csv(filepath, names)


def preprocess(breast):
    breast.info()  #查看是否有缺失值、重复数据
    # 该数据集存在字符串类型数据'?'
    # 将'?'转换成nan
    breast = breast.replace(to_replace='?', value=np.nan)
    # 将nan所在的行删除
    breast = breast.dropna()

    # 特征值是除了class列以外的所有数据
    features = breast.drop('Class', axis=1)
    # 目标值是class这一列
    targets = breast['Class']

    return features, targets


if __name__ == '__main__':
    #1、数据获取
    # 癌症数据路径
    filepath = '机器学习/逻辑回归/data/breast-cancer-wisconsin.data'
    # 癌症的每一项特征名
    names = [
        'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
        'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
        'Mitoses', 'Class'
    ]

    #加载数据
    breast = load_data(filepath, names)
    # 查看唯一值，Class这列代表的是否得癌症，使用.unique()函数查看该列有哪些互不相同的值
    unique = breast['Class'].unique()  #只有两种情况，是二分类问题，2代表良性，4代表恶性

    #2、数据处理
    features, targets = preprocess(breast)

    #3、划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25)

    #4、特征工程
    # 接收标准化方法
    transfer = StandardScaler()
    # 对训练的特征值x_train提取特征并标准化处理
    x_train = transfer.fit_transform(x_train)
    # 对测试的特征值x_test标准化处理
    x_test = transfer.transform(x_test)

    #5、逻辑回归预测
    # 接收逻辑回归方法
    logist = LogisticRegression()
    # penalty=l2正则化；tol=0.001损失函数小于多少时停止；C=1惩罚项，越小惩罚力度越小，是岭回归的乘法力度的分之一
    # 训练
    logist.fit(x_train, y_train)
    # 预测
    y_predict = logist.predict(x_test)
    # 评分法计算准确率
    accuracy = logist.score(x_test, y_test)

    #6、准确率和召回率
    # classification_report()
    # 参数(真实值,预测值,labels=None,target_names=None)
    # labels：class列中每一项，如该题的2和4，给它们取名字
    # target_names：命名

    # 计算准确率和召回率
    res = classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性'])
    # precision准确率；recall召回率；综合指标F1-score；support：预测的人数
    print(res)