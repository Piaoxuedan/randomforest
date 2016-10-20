import numpy as np
from numpy import *
from ClassTree import ClassificationTree
from ClassTreeBagging import TreeBagger
from ClassForest import RandomForest
import csv
import kafang

useful_feat = kafang.main()


tree = ClassificationTree()
bag = TreeBagger(n_trees=15)
forest = RandomForest(n_trees=15)

dataset = genfromtxt('/Users/dan/Desktop/Data/N-train.csv', delimiter=',')[1:]
train_list = list()
for i in useful_feat:
    train_list.append([x[i] for x in dataset])
train_pre = np.array(train_list)
train = np.array(mat(train_pre).T)
print (len(train))

target_list = genfromtxt('/Users/dan/Desktop/Data/target-2.csv',delimiter= ',')[1:]
target = np.array(target_list)

#test_list = genfromtxt('/Users/dan/Documents/Python/RandomForest-gh-pages/N-test.csv', delimiter=',' )[1:]
#test = np.array(test_list)
test_dataset = genfromtxt('/Users/dan/Desktop/Data/N-test.csv', delimiter=',' )[1:]
test_list = list()
for i in useful_feat:
    test_list.append([x[i] for x in test_dataset])
test_pre = np.array(test_list)
test = np.array(mat(test_pre).T)
print (len(test))

tree.train(train,target)
print("Accuracy of the simple tree on dataset is %f" % tree.evaluate(train, target))

tree.describe()
#write to json
js = tree.to_json("test_json.json")

#load from json
tree = tree.from_json("test_json.json")

#check it is the same
tree.describe()
print("Accuracy of the reloaded tree on iris dataset is %f" % tree.evaluate(train, target))
# Cross validation of a tree
tree.cross_val(train, target, folds=5)

# Tree bag
bag.train(train, target)
print("Accuracy of the bagged forest on dataset is %f" % bag.evaluate(train, target))
# Cross validation of a tree bag
bag.cross_val(train, target, folds=5)

# Random forest
forest.train(train, target)
print("Accuracy of the random forest on dataset is %f" % forest.evaluate(train, target))


results = forest.predict(test)
print  (len(results))
with open("/Users/dan/Desktop/Data/test-results.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageId", "Label"])
    for i, predicted_digit in enumerate(results):
        writer.writerow((i + 1, int(predicted_digit)))