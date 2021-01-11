import numpy as np
import os
from random import randrange


def safe_float(number):
    try:
        return float(number)
    except:
        return number


# data
data_root = "./data_regress/"
data_files = []
dataSet = []
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
predict_name = 'MEDV'

for root, dirs, files in os.walk(data_root):
    for file in files:
        data_files.append(os.path.join(data_root, file))
print(data_files)
for file in data_files:
    with open(file, 'r') as f:
        lines = list(f)
        for line in lines:
            a = line.split()
            a = np.array(a)
            a = list(map(safe_float, a))
            dataSet.append(a)
print(dataSet)
dataSet = np.array(dataSet)  # 这里将列表转换为数组
print('dataSet: ', dataSet.shape)

# 将数据集随机分成n块，其中一块是测试集，其他n-1是训练集
def getTrainTest(dataSet, n_folds):
    train_size = int(len(dataSet) / n_folds) * (n_folds - 1)
    dataSet_copy = list(dataSet)
    train = []
    for i in range(n_folds - 1):
        while len(train) < train_size:  # 这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立
            index = randrange(len(dataSet_copy))
            train.append(dataSet_copy.pop(index))  # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
    test = dataSet_copy
    return train, test


def getDataSet():
    return dataSet, feature_names, predict_name
