from sklearn import datasets
from sklearn.model_selection import train_test_split
from math import log

from BinaryTree import *
from plt_confusion_matrix import plt_confusion_matrix
from tree_tools import *
from read_classify_data import getDataSet, getTrainTest

# data
dataSet, feature_names, class_name, classNum = getDataSet()
trainData, testData = getTrainTest(dataSet, 5)
trainData = np.array(trainData)
testData = np.array(testData)
print('trainData: ', trainData.shape)
print('testData: ', testData.shape)

ENTROPY = False  # False就用gini


def entropy(data_set):
    count = len(data_set)
    label_counts = {}

    # Count the number of each category in the dataset
    for row in data_set:
        label = row[-1]
        if label not in label_counts.keys():
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / count
        entropy -= prob * log(prob, 2)
    return entropy


def gini(data_set):
    count = len(data_set)
    label_counts = {}

    # Count the number of each category in the dataset
    for row in data_set:
        label = row[-1]
        if label not in label_counts.keys():
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    impurity = 1.0
    for key in label_counts:
        prob = float(label_counts[key]) / count
        impurity -= prob * prob
    return impurity


# Select the feature with the highest gain base on entropy
def choose_best_feature_entropy(data_set):
    feature_count = len(data_set[0]) - 1
    # The original entropy of the data set
    base_entropy = entropy(data_set)
    # Maximum gain
    best_gain = 0.0
    # Maximum gain features
    best_feature = [-1, -1]  # [id, value]

    # Go through each feature
    for i in range(feature_count):
        feature = [example[i] for example in data_set]
        feature_value_set = set(feature)
        # Calculated gain
        for value in feature_value_set:
            left, right = split_data_set(data_set, i, value)
            prob = len(left) / float(len(data_set))
            gain = base_entropy - prob * entropy(left) - (1 - prob) * entropy(right)
            # Compare
            if gain > best_gain:
                best_gain = gain
                best_feature = [i, value]
    return best_feature


# Select the feature with the highest gain base on gini
def choose_best_feature_gini(data_set):
    feature_count = len(data_set[0]) - 1
    base_gini = gini(data_set)
    best_gain = 0.0
    best_feature = [-1, -1]

    for i in range(feature_count):
        feature = [example[i] for example in data_set]
        feature_value_set = set(feature)
        for value in feature_value_set:
            left, right = split_data_set(data_set, i, value)
            prob = len(left) / float(len(data_set))
            gain = base_gini - prob * gini(left) - (1 - prob) * gini(right)
            if gain > best_gain:
                best_gain = gain
                best_feature = [i, value]
    return best_feature


def count_value(rows):
    count = {}
    label = ""
    # takes whole dataset in as argument
    for row in rows:
        # traverse on each datapoint
        label = row[-1]
        # labels are in the last column
        # if label is not even once come initialise it
        if label not in count:
            count[label] = 0
        # increase the count of present label by 1
        count[label] += 1
    return label, count[label]


# Create decision tree
def create_division_tree(data_set, feature_names, parent, left=False, right=False):
    class_list = [example[-1] for example in data_set]

    # Return leaf node if all data are in the same class
    if class_list.count(class_list[0]) == len(class_list):
        label, num = count_value(data_set)
        leaf = Leaf(label, num)
        node = BinaryTree(leaf)
        if left:
            parent.insertLeft(node)
        elif right:
            parent.insertRight(node)
        return node
    else:
        if ENTROPY:
            best_feature = choose_best_feature_entropy(data_set)
        else:
            best_feature = choose_best_feature_gini(data_set)
        print("best_feature:", feature_names[best_feature[0]], best_feature[1])
        leftData, rightData = split_data_set(data_set, best_feature[0], best_feature[1])

        # Creates the current decision node
        tempNode = BinaryTree(DecisionNode(best_feature[0], best_feature[1], feature_names[best_feature[0]]))
        if left:  # If the current DecisionNode is the left child of the previous node, add to left
            parent.insertLeft(tempNode)
        elif right:  # right
            parent.insertRight(tempNode)
        create_division_tree(leftData, feature_names, tempNode, left=True)  # iteration
        create_division_tree(rightData, feature_names, tempNode, right=True)
        return tempNode


#  train
my_tree = create_division_tree(trainData, feature_names, None)

confusion = np.zeros((classNum, classNum))
#  test
true_num = 0
for data in testData:
    pre, real = test(my_tree, data)
    if pre == real:
        true_num += 1
    confusion[class_name[real]][class_name[pre]] += 1
print(confusion)
acc = true_num / len(testData)
ACC = str("%.03f" % acc)
print("accuracy=" + ACC)

# save results
if ENTROPY:
    save_path = str('./figures/my_classifier_entropy_confusion_' + ACC + '.png')
else:
    save_path = str('./figures/my_classifier_gini_confusion_' + ACC + '.png')
plt_confusion_matrix(confusion, class_name.keys(), save_path)

if ENTROPY:
    save_path = str('./figures/my_classifier_entropy_tree_' + ACC + '.gv')
else:
    save_path = str('./figures/my_classifier_gini_tree_' + ACC + '.gv')
my_tree.print_tree(save_path=save_path)
