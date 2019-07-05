import warnings
warnings.simplefilter("ignore")

import numpy
import pandas
from numpy import log2 as log

def visualize(tree, graph, prev, stat, count):
    for k,v in tree.items():
        check = False
        if k is 'Outlook' or k is 'Temperature' or k is 'Wind' or k is 'Humidity':
            check = True
        if v == 'Yes' or v == 'No':
            graph.edge(prev, v + str(count), label = k)
            count += 1
        if prev != '' and check == True:
            graph.edge(prev, k, label = stat)
        if check == True:
            prev = k
        if(type(v) is dict):
            stat = k
            visualize(v, graph, prev, stat, count)

# Expected information (entropy) needed to classify a tuple in D:
def get_entropy(dataset):
    feature = dataset.keys()[-1]
    entropy = 0
    values = dataset[feature].unique()
    for value in values:
        fraction = dataset[feature].value_counts()[value] / len(dataset[feature])
        entropy += -fraction * numpy.log(fraction)
    return entropy
  
# Information needed (after using A to split D into v partitions) to classify D:
def get_entropy_features(dataset, attribute):
  eps = numpy.finfo(float).eps
  feature = dataset.keys()[-1]
  tVar = dataset[feature].unique()
  variables = dataset[attribute].unique()
  final_entropy = 0
  for v in variables:
      entropy = 0
      for target_variable in tVar:
          num = len(dataset[attribute][dataset[attribute]==v][dataset[feature] == target_variable])
          d = len(dataset[attribute][dataset[attribute]==v])
          fraction = num / (d+eps)
          entropy += -fraction * log(fraction + eps)
      final_entropy += -(d/len(dataset)) * entropy
  return abs(final_entropy)

# Information gained by branching on feature
def find_partition(dataset):
    container = []
    for key in dataset.keys()[:-1]:
        container.append(get_entropy(dataset) - get_entropy_features(dataset,key))
    return dataset.keys()[:-1][numpy.argmax(container)]

def buildTree(dataset, tree=None):

    # find the best split point
    node = find_partition(dataset)
    values = numpy.unique(dataset[node])
    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
    for value in values:
        subtable = dataset[dataset[node] == value].reset_index(drop=True)
        val, counts = numpy.unique(subtable['PlayTennis'], return_counts=True)                        
        
        if len(counts) == 1:
            tree[node][value] = val[0]
        else: # recursively make the tree
            tree[node][value] = buildTree(subtable)
                   
    return tree
  

def predict(inst,tree):
    for nodes in tree.keys():        
        value = inst[nodes]
        tree = tree[nodes].get(value)
        # if(tree == None):
        #     tree = inst['PersonalLoan']
        predictedVal = 0
            
        if type(tree) is dict:
            predictedVal = predict(inst, tree)
        else:
            predictedVal = tree
            break                     
    return predictedVal


dataset = pandas.read_csv("SampleData.csv")
dataset2 = pandas.read_csv("SampleData.csv")
dataset2 = dataset.loc[:,['Outlook','Temperature','Humidity','Wind', 'PlayTennis']]

categorical_cols = ['Outlook','Temperature','Humidity','Wind']

for column in categorical_cols:
    dataset[column] = pandas.factorize(dataset[column])[0]

dataset['PlayTennis'] = dataset['PlayTennis'].replace('Yes', 1)
dataset['PlayTennis'] = dataset['PlayTennis'].replace('No', -1)

dataset = dataset.loc[:,['Outlook','Temperature','Humidity','Wind', 'PlayTennis']]

portion = int(len(dataset) * 0.7)
train = dataset[:portion]
test = dataset[portion:len(dataset)]

tree = buildTree(train)
import pprint
tree2 = buildTree(dataset2[:portion])
pprint.pprint(tree2)

from graphviz import Digraph
u = Digraph('unix', filename='sample_graph', format = 'png')
u.attr(size='6,6')
u.node_attr.update(color='lightblue2', style='filled')
visualize(tree2, u, "", "", 0)
u.view()

def get_confusion_matrix(tree, test, val):
    testInput = []
    testResult = []
    for row in test.itertuples(index=False, name=None):
        data = {'Outlook':row[0], 'Temperature':row[1], 'Humidity':row[2], 'Wind':row[3] }
        testInput.append(data)
        testResult.append({'PlayTennis':row[4] })
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(testInput)):
        result = predict(testInput[i], tree)
        if result == testResult[i]['PlayTennis']:
            if testResult[i]['PlayTennis'] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if testResult[i]['PlayTennis'] == 1:
                fp += 1
            else:
                fn += 1
    
    matrix = [[0 for i in range(2)] for j in range(2)]
    matrix[1][1] = tp
    matrix[0][1] = fn
    matrix[1][0] = fp
    matrix[0][0] = tn
    print("Test set size =", val)
    print("  Accuracy =", (tp + tn) / (fp + fn + tp + tn))
    print("  Precision =", tp / (tp + fp))
    print("  Recall =", tp / (tp + fn))
    return matrix

matrix = get_confusion_matrix(tree, test, 0.7)

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
fig, ax1 = plot_confusion_matrix(numpy.array(matrix))
plt.show()
 

# Show Heatmap XX
import seaborn as sns
corr = dataset.corr()
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()
 

# Show Heatmap XY
corr = pandas.DataFrame(dataset[['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']].corr()['PlayTennis'][:])
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])
plt.show()
 

x = []
y = []

for i in range(5, 10):
    x.append(i * 0.1)
    portion = int(len(dataset) * 0.1 * i)
    train = dataset[:portion]
    test = dataset[portion:len(dataset)]

    tree = buildTree(train)
    matrix = get_confusion_matrix(tree, test, i * 0.1)
    accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
    y.append(accuracy)
    
plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=9, label = "Random Subset")
plt.show()
 

