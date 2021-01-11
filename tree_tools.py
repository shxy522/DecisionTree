import numpy as np


def split_data_set(data_set, axis, value):
    """根据指定条件分割数据集"""
    # 划分后的新数据集
    left = []
    right = []

    for data in data_set:
        if data[axis] <= value:
            left.append(data)
        else:
            right.append(data)

    return np.array(left), np.array(right)


#  测试一个数据
def test(treeNode, data):
    """遍历决策树对测试数据进行分类"""
    if type(treeNode.key).__name__ == 'DecisionNode':  # 如果不是叶子节点 递归
        if data[treeNode.key.feature_id] <= treeNode.key.feature_value:
            return test(treeNode.leftChild, data)
        else:
            return test(treeNode.rightChild, data)
    else:  # 如果是叶子节点，判断分类是否正确
        return treeNode.key.value, data[-1]
