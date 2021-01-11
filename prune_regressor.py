from BinaryTree import *
from tree_tools import *


def isTree(node):
    if type(node.key).__name__ == 'DecisionNode':
        return True  # if it is tree, return True


# make tree into leaf
def getLeaf(tree):
    if isTree(tree.rightChild):
        tree.rightChild = getLeaf(tree.rightChild)
    if isTree(tree.leftChild):
        tree.leftChild = getLeaf(tree.leftChild)

    value = (tree.leftChild.key.value + tree.rightChild.key.value) / 2
    num = tree.leftChild.key.num + tree.rightChild.key.num
    leaf = Leaf(value, num)
    return leaf  # return leaf node


# use testData to prune
def prune(tree, testData):
    # 1. If testData is empty, the tree would be made into a leaf.
    if testData.shape[0] == 0:
        node = BinaryTree(getLeaf(tree))
        return node

    # 2. Split data base on the tree
    if isTree(tree.leftChild) or isTree(tree.rightChild):
        leftData, rightData = split_data_set(testData, tree.key.feature_id, tree.key.feature_value)

    # 3. Iteration
    if isTree(tree.leftChild):
        tree.leftChild = prune(tree.leftChild, leftData)  # left
    if isTree(tree.rightChild):
        tree.rightChild = prune(tree.rightChild, rightData)  # right

    # 4. If the tree here is a leaf, try to merge if it is necessary
    if not isTree(tree.leftChild) and not isTree(tree.rightChild):
        leftData, rightData = split_data_set(testData, tree.key.feature_id, tree.key.feature_value)

        # compare the error
        # 4-1. get current error
        error_left = 0
        error_right = 0
        if leftData.shape[0] != 0:
            error_left = np.sum(np.power(leftData[:, -1] - tree.leftChild.key.value, 2))
        if rightData.shape[0] != 0:
            error_right = np.sum(np.power(rightData[:, -1] - tree.rightChild.key.value, 2))
        errorNoMerge = np.sum(error_left) + np.sum(error_right)
        # 4-2. get the error after merge
        treemean = (tree.leftChild.key.value + tree.rightChild.key.value) / 2
        errorMerge = np.sum(np.power(testData[:, -1] - treemean, 2))
        # 4-3. compare
        if errorMerge < errorNoMerge:  # to merge
            print("merging")
            node = BinaryTree(getLeaf(tree))
            return node
        else:
            return tree  # return the tree
    return tree  # return the tree
