# coding:utf-8
# python 2.7
from numpy import *
from ClassTree import ClassificationTree
from ClassTreeBagging import TreeBagger
from ClassForest import RandomForest
import csv
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np



if __name__ == "__main__":
    result_mat = np.zeros((5,12),dtype=np.int)
    tree = ClassificationTree("entropy")
    bag = TreeBagger(n_trees=15)
    forest = RandomForest(n_trees=15)
    dataset = genfromtxt('/Users/dan/Desktop/Data/N-train.csv', delimiter=',')[1:]
    train = np.array(dataset)
    print len(train)

    target_list = genfromtxt('/Users/dan/Desktop/Data/target-2.csv', delimiter=',')[1:]
    target = np.array(target_list)

    test_dataset = genfromtxt('/Users/dan/Desktop/Data/N-test.csv', delimiter=',')[1:]
    test = np.array(test_dataset)
    print (len(test))

    tree.train(train, target)
    print("Accuracy of the simple tree on dataset is %f" % tree.evaluate(train, target))

    tree.describe()
    # write to json
    js = tree.to_json("test_json.json")

    # load from json
    tree = tree.from_json("test_json.json")

    # check it is the same
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
    nodes = forest.train(train, target)
    print("Accuracy of the random forest on dataset is %f" % forest.evaluate(train, target))

    results = forest.predict(test)

    result_mat[0] = np.array(results)
    print result_mat.shape
    print 'jieguojuzhen'
    for q in range(4):
        tree = ClassificationTree("entropy")
        bag = TreeBagger(n_trees=15)
        forest = RandomForest(n_trees=15)
        nodes_2 = list(set(nodes))

        dataset = genfromtxt('/Users/dan/Desktop/Data/N-train.csv', delimiter=',')[1:]
        train_list = list()
        print len(nodes_2)
        print '特征长度'
        for i in nodes_2:
            train_list.append([x[i] for x in dataset])
        train_pre = np.array(train_list)
        train = np.array(mat(train_pre).T)
        print train.shape
        print '训练集的形状'

        target_list = genfromtxt('/Users/dan/Desktop/Data/target-2.csv',delimiter= ',')[1:]
        target = np.array(target_list)

        test_dataset = genfromtxt('/Users/dan/Desktop/Data/N-test.csv', delimiter=',' )[1:]
        test_list = list()
        for i in nodes_2:
            test_list.append([x[i] for x in test_dataset])
        test_pre = np.array(test_list)
        test = np.array(mat(test_pre).T)
        print test.shape
        print '测试集形状'

        tree.train(train,target)
        #print("Accuracy of the simple tree on dataset is %f" % tree.evaluate(train, target))

        tree.describe()
        js = tree.to_json("test_json.json")
        tree = tree.from_json("test_json.json")
        tree.describe()
        #print("Accuracy of the reloaded tree on iris dataset is %f" % tree.evaluate(train, target))
        tree.cross_val(train, target, folds=5)
        bag.train(train, target)
        #print("Accuracy of the bagged forest on dataset is %f" % bag.evaluate(train, target))
        # Cross validation of a tree bag
        bag.cross_val(train, target, folds=5)

        # Random forest
        nodes = forest.train(train, target)
        print("Accuracy of the random forest on dataset is %f" % forest.evaluate(train, target))

        results = forest.predict(test)
        print results

        result_mat[q+1] = np.array(results)
    print result_mat[1]
    print '理论上最好的一次结果'

    results_final = list()
    for p in range(12):
        p_dict = dict()
        p_result = result_mat[:, p] #第p个测试集的所有结果
        print p_result #numpy.ndarray
        classes, counts = np.unique(p_result, return_counts=True)
        votemax = np.amax(counts)
        max_index = np.argwhere(counts==votemax)
        p_vote = classes[max_index[0][0]]
        results_final.append(p_vote)
    print results_final
    print '最终结果'


    with open("/Users/dan/Desktop/Data/test-results.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "Label"])
        for i, predicted_digit in enumerate(results_final):
            writer.writerow((i + 1, int(predicted_digit)))
