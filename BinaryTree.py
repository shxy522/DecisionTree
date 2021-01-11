from graphviz import Digraph
import uuid
from random import sample


class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
        self.dot = Digraph(comment='Binary Tree')

    def insertLeft(self, newNode):
        if self.leftChild is None:
            self.leftChild = newNode
        else:
            t = newNode
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild is None:
            self.rightChild = newNode
        else:
            t = newNode
            t.rightChild = self.rightChild
            self.rightChild = t

    # Using Graphviz to realize binomial visualization
    def print_tree(self, save_path='./Binary_Tree.gv', label=True):

        # colors for labels of nodes
        colors = ['skyblue', 'tomato', 'orange', 'purple', 'green', 'yellow', 'pink', 'red']

        # Draws a binary tree with a node as its root
        def print_node(node, node_tag):
            # The node color
            color = sample(colors, 1)[0]
            if node.leftChild is not None:
                if type(node.leftChild.key).__name__ == 'DecisionNode':  # DecisionNode
                    left_tag = str(uuid.uuid1())  # Data for the left node
                    self.dot.node(left_tag,
                                  str(node.leftChild.key.feature_name + " <= " + str(
                                      node.leftChild.key.feature_value) + " ? "),
                                  style='filled', color=color)  # left
                    label_string = 'Y' if label else ''  # Whether to label the connector indicates a left subtree
                    self.dot.edge(node_tag, left_tag, label=label_string)  # The line between the left child and parent
                    print_node(node.leftChild, left_tag)
                else:  # leaf
                    left_tag = str(uuid.uuid1())
                    self.dot.node(left_tag, str(
                        "samples = " + str(node.leftChild.key.num) + "\n value = " + str(node.leftChild.key.value)),
                                  style='filled', color=color)
                    label_string = 'Y' if label else ''
                    self.dot.edge(node_tag, left_tag, label=label_string)
                    print_node(node.leftChild, left_tag)

            if node.rightChild is not None:
                if type(node.rightChild.key).__name__ == 'DecisionNode':
                    right_tag = str(uuid.uuid1())
                    self.dot.node(right_tag, str(node.rightChild.key.feature_name + " <= " + str(
                        node.rightChild.key.feature_value) + " ? "), style='filled', color=color)
                    label_string = 'N' if label else ''
                    self.dot.edge(node_tag, right_tag, label=label_string)
                    print_node(node.rightChild, right_tag)
                else:
                    right_tag = str(uuid.uuid1())
                    self.dot.node(right_tag, str(
                        "samples = " + str(node.rightChild.key.num) + "\n value = " + str(node.rightChild.key.value)),
                                  style='filled', color=color)
                    label_string = 'N' if label else ''
                    self.dot.edge(node_tag, right_tag, label=label_string)
                    print_node(node.rightChild, right_tag)

        if type(self.key).__name__ == 'DecisionNode':
            root_tag = str(uuid.uuid1())  # root
            self.dot.node(root_tag, str(self.key.feature_name + "<=" + str(self.key.feature_value) + "?"),
                          style='filled', color=sample(colors, 1)[0])  # Create the root node
            print_node(self, root_tag)

        self.dot.render(save_path)


class DecisionNode:
    def __init__(self, feature_id, feature_value, feature_name):
        # question object stores col and val variables regarding the question of that node
        self.feature_id = feature_id
        self.feature_value = feature_value
        self.feature_name = feature_name


class Leaf:
    def __init__(self, value, num):
        # stores unique labels and their values in predict
        self.value = value
        self.num = num
        self.predict = {self.value, self.num}
