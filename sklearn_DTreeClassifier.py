from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from plt_confusion_matrix import plt_confusion_matrix

ENTROPY = True  # False is using gini

from read_classify_data import getDataSet, getTrainTest

# data
dataSet, feature_names, class_name, classNum = getDataSet()
trainData, testData = getTrainTest(dataSet, 5)
trainData = np.array(trainData)
testData = np.array(testData)
x_train = trainData[:, :-1]
y_train = trainData[:, -1]
x_test = testData[:, :-1]
y_test = testData[:, -1]
print('trainData: ', trainData.shape)
print('testData: ', testData.shape)

if ENTROPY:
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0)
else:
    classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=0)

plt.figure(figsize=(12, 8))
# fit the model
tree.plot_tree(classifier.fit(x_train, y_train))
if ENTROPY:
    save_path = str('./figures/sklearn_classifier_entropy_tree.pdf')
else:
    save_path = str('./figures/sklearn_classifier_gini_tree.pdf')
plt.savefig(save_path, bbox_inches='tight', format='pdf')
plt.clf()

# predict
y_pred = classifier.predict(x_test)
print('Model accuracy score index: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
if ENTROPY:
    save_path = str(
        './figures/sklearn_classifier_entropy_confusion_{0:0.4f}.png'.format(accuracy_score(y_test, y_pred)))
else:
    save_path = str('./figures/sklearn_classifier_gini_confusion_{0:0.4f}.png'.format(accuracy_score(y_test, y_pred)))
plt_confusion_matrix(cm, class_name.keys(), save_path)
