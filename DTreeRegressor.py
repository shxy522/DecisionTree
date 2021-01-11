from sklearn import datasets
from sklearn.model_selection import train_test_split

from BinaryTree import *
from tree_tools import *
from prune_regressor import prune
from read_regress_data import getDataSet, getTrainTest

# data
dataSet, feature_names, predict_name = getDataSet()
trainData, testData = getTrainTest(dataSet, 5)
trainData = np.array(trainData)
testData = np.array(testData)
print('trainData: ', trainData.shape)
print('testData: ', testData.shape)

# The average value of leaf node data set is the predicted value
# return value, dataNum
def count_value(data_set):
    result = data_set[:, -1]
    value = np.mean(result)
    return value, len(result)


# The sum of squares of the difference between the estimated value and the true value
def rss(data_set):
    result = data_set[:, -1]
    return np.var(result) * (np.shape(data_set)[0])


# ops(x,y):
# x--Minimum error reduction
# y--Minimum number of samples after classification
def choose_best_split(data_set, ops=(1, 4)):
    tols = ops[0]
    toln = ops[1]

    # if all data are the same, return none
    if len(set(data_set[:, -1])) == 1:
        return None, None

    m, n = np.shape(data_set)
    s = rss(data_set)

    best_s = np.inf
    best_index = 0
    best_value = 0

    # Go through each of the features
    for feat_index in range(n - 1):
        # Go through all values for the current feature
        for value in np.unique(data_set[:, feat_index]):
            left, right = split_data_set(data_set, feat_index, value)
            # If the number of samples is small after classification, exit the loop
            if np.shape(left)[0] < toln or np.shape(right)[0] < toln:
                continue
            # Calculate a new error
            new_s = rss(left) + rss(right)

            # Update minimum error
            if new_s < best_s:
                best_index = feat_index
                best_value = value
                best_s = new_s

    # If the error reduction is not significant, exit
    if (s - best_s) < tols:
        return None, None

    # If the slicing data set is small, exit
    left, right = split_data_set(data_set, best_index, best_value)
    if left.shape[0] < toln or right.shape[0] < toln:
        return None, None

    return best_index, best_value


def create_division_tree(data_set, feature_names, parent, left=False, right=False, ops=(1, 4)):
    best_index, best_value = choose_best_split(data_set, ops=ops)
    if best_index is None:
        value, num = count_value(data_set)
        node = BinaryTree(Leaf(value, num))
        if left:
            parent.insertLeft(node)
        elif right:
            parent.insertRight(node)
        return node
    else:
        leftData, rightData = split_data_set(data_set, best_index, best_value)
        # create DecisionNode
        tempNode = BinaryTree(DecisionNode(best_index, best_value, feature_names[best_index]))
        if left:  # If the current DecisionNode is the left child of the previous node, add to left
            parent.insertLeft(tempNode)
        elif right:  # right
            parent.insertRight(tempNode)
        create_division_tree(leftData, feature_names, tempNode, left=True, ops=ops)  # Iteration
        create_division_tree(rightData, feature_names, tempNode, right=True, ops=ops)

    return tempNode


# train
my_tree = create_division_tree(trainData, feature_names, None, ops=(10, 5))
# prune
my_tree = prune(my_tree, testData)

# test
sse = 0
sst = 0
mean_real = np.mean(testData[:, -1])
for data in testData:
    predict_value, real = test(my_tree, data)
    sse += ((real - predict_value) ** 2)
    sst += ((real - mean_real) ** 2)
r_square = 1 - sse / sst

# save results
SSE = str("%.03f" % sse)
R2 = str("%.03f" % r_square)
print("SSE:" + SSE)
print("R2:" + R2)
save_path = str('./figures/my_regressor_tree_SSE_' + SSE + '_R2_' + R2 + '.gv')
my_tree.print_tree(save_path=save_path)
