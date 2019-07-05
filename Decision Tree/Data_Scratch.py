import warnings
warnings.simplefilter("ignore")

import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pprint
from numpy import log2 as log
import numpy
import pandas

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
        val, counts = numpy.unique(subtable['PersonalLoan'], return_counts=True)                        
        
        if len(counts) == 1:
            tree[node][value] = val[0]
        else: # recursively make the tree
            tree[node][value] = buildTree(subtable)
                   
    return tree
  

def predict(inst,tree):
    for nodes in tree.keys():        
        value = inst[nodes]
        tree = tree[nodes].get(value)
        if(tree == None):
            tree = inst['PersonalLoan']
        predictedVal = 0
            
        if type(tree) is dict:
            predictedVal = predict(inst, tree)
        else:
            predictedVal = tree
            break                     
    return predictedVal

dataset = pandas.read_csv("Data.csv")

categorical_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
                    'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']

for column in categorical_cols:
    if(column == 'Age' or column == 'Experience' or column == 'Income' or column == 'CCAvg' or column == 'Mortage'):
        min = dataset[column].describe()['min']
        max = dataset[column].describe()['max']
        times = 40
        val = min
        increment = (max - min) / times
        for i in range(0, times):
            dataset[column] = numpy.where(dataset[column].between(
                val, val + increment), val, dataset[column])
            val += increment

dataset = dataset.loc[:, ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
                          'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard', 'PersonalLoan']]

portion = int(len(dataset) * 0.7)
train = dataset[:portion]
test = dataset[portion:len(dataset)]

tree = buildTree(train)
pprint.pprint(tree)

def get_confusion_matrix(tree, test, val):
    testInput = []
    testResult = []
    for row in test.itertuples(index=False, name=None):
        data = {
            'Age': row[0],
            'Experience': row[1],
            'Income': row[2],
            'Family': row[3],
            'CCAvg': row[4],
            'Education': row[5],
            'Mortgage': row[6],
            'SecuritiesAccount': row[7],
            'CDAccount': row[8],
            'Online': row[9],
            'CreditCard': row[10],
            'PersonalLoan': row[11]
        }
        testInput.append(data)
        testResult.append({'PersonalLoan': row[11]})
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(testInput)):
        result = predict(testInput[i], tree)
        if result == testResult[i]['PersonalLoan']:
            if testResult[i]['PersonalLoan'] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if testResult[i]['PersonalLoan'] == 1:
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

fig, ax1 = plot_confusion_matrix(numpy.array(matrix))
plt.show()

# Show Heatmap A
corr = dataset.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()

# Show Heatmap B
corr = pandas.DataFrame(dataset[['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
                                 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard', 'PersonalLoan']].corr()['PersonalLoan'][:])
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=['Age', 'Experience', 'Income', 'Family', 'CCAvg',
                                                         'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard', 'PersonalLoan'])
plt.show()

x = []
y = []

for i in range(5, 10):
    x.append(i * 0.1)
    portion = int(len(dataset) * 0.1 * i)
    train = dataset[:portion]
    test = dataset[portion:len(dataset)]

    tree = buildTree(train)
    matrix = get_confusion_matrix(tree, test, 0.1 * i)
    accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
    y.append(accuracy)

plt.plot(x, y, color='green', linestyle='dashed', linewidth=3, marker='o',
         markerfacecolor='blue', markersize=9, label="Random Subset")
plt.show()
