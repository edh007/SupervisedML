import warnings
warnings.simplefilter("ignore")

import sklearn
import pandas
import reports

dataset = pandas.read_csv("Data.csv")

#categorical_cols = ['ID','Age','Experience','Income','ZIPCode','Family','CCAvg','Education','Mortgage','PersonalLoan','SecuritiesAccount','CDAccount','Online','CreditCard']
categorical_cols = ['Age','Experience','Income','ZIPCode','Family','CCAvg','Education','Mortgage','SecuritiesAccount','CDAccount','Online','CreditCard']

# for column in categorical_cols:
#     dataset[column] = pandas.factorize(dataset[column])[0]

dataset['PersonalLoan'] = dataset['PersonalLoan'].replace(1, 1)
dataset['PersonalLoan'] = dataset['PersonalLoan'].replace(0, -1)

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, train_size = 0.7, random_state = 1, shuffle = False)

from sklearn.tree import DecisionTreeClassifier, export_graphviz

dt = DecisionTreeClassifier()
decision_tree_binary_classifier = dt.fit(train[categorical_cols], train.PersonalLoan)

import numpy as np
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
import os
from graphviz import Source

dot_data = export_graphviz(decision_tree_binary_classifier, out_file = None, feature_names=np.array(categorical_cols),
    class_names=['Loan', 'Not Loan'], filled=True, rounded=True, special_characters=True)
graph = Source(dot_data)
graph.format = 'png'
graph.render('dtree_render', view=True)


from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

test_output = decision_tree_binary_classifier.predict(test[categorical_cols])

report = classification_report(
    test.PersonalLoan,
    test_output,
    target_names=['NotLoan', 'Loan']
)

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        if len(t) < 2: continue
        targetNum = -1
        if isFloat(t[1]) == True:
            targetNum = 0
        else:
            targetNum = 1
        classes.append(t[targetNum])
        v = [float(x) for x in t[targetNum + 1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()


plot_classification_report(report)
 

fig, ax1 = plot_confusion_matrix((confusion_matrix(test.PersonalLoan, test_output)))
plt.show()
 

# Show Heatmap A
import seaborn as sns
corr = dataset.corr()
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()
 

# Show Heatmap B
corr = pandas.DataFrame(dataset[['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard' ,'PersonalLoan']].corr()['PersonalLoan'][:])
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard' ,'PersonalLoan'])
plt.show()
 

#show Number of training sample vs accuracy
x = []
y = []

for i in range(5, 10):
    train, test = train_test_split(dataset, train_size = 0.1 * i, random_state = 0, shuffle = False)
    dt = DecisionTreeClassifier()
    decision_tree_binary_classifier = dt.fit(train[categorical_cols], train.PersonalLoan)
    test_output = decision_tree_binary_classifier.predict(test[categorical_cols])
    report = classification_report(test.PersonalLoan, test_output, target_names=['NotLoan', 'Loan'])
    x.append(i * 0.1)
    accuracy = decision_tree_binary_classifier.score(test[categorical_cols], test.PersonalLoan)
    y.append(accuracy)

plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=9, label = "Random Subset")
plt.show()
