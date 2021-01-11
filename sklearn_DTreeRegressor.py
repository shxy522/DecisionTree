import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz
from read_regress_data import getDataSet, getTrainTest

# data
dataSet, feature_names, predict_name = getDataSet()
trainData, testData = getTrainTest(dataSet, 5)
trainData = np.array(trainData)
testData = np.array(testData)
x_train = trainData[:, :-1]
y_train = trainData[:, -1]
x_test = testData[:, :-1]
y_test = testData[:, -1]
print('trainData: ', trainData.shape)
print('testData: ', testData.shape)

# Fit regression model
tree = DecisionTreeRegressor(max_depth=10)
tree.fit(x_train, y_train)

# Predict
y_prediect = tree.predict(x_test)
sse = np.sum((y_test - y_prediect) ** 2)
sst = np.sum((y_test - np.mean(y_test)) ** 2)
r_square = 1 - sse / sst
SSE = str("%.03f" % sse)
R2 = str("%.03f" % r_square)
print("SSE:" + SSE)
print("R2:" + R2)
save_path = str('./figures/sklearn_regressor_tree_SSE_' + SSE + '_R2_' + R2 + '.gv')

export_graphviz(tree, out_file='./figures/tree.dot', class_names=["malignant", "benign"],
                feature_names=feature_names,
                impurity=False, filled=True)
with open('./figures/tree.dot') as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph, format="pdf")  # 保存到pdf矢量图
dot.render(save_path)
