# coding:utf-8
# python 2.7
import numpy as np
from numpy import *
from ClassTree import ClassificationTree
from ClassTreeBagging import TreeBagger
from ClassForest import RandomForest
import csv
from sklearn.preprocessing import MinMaxScaler


import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Smote:
    def __init__(self,samples,N=10,k=3):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
        self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print 'neighbors',neighbors
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

if __name__ == "__main__":
    result_mat = np.zeros((5,12),dtype=np.int)
    for q in range(5):
        #useful_feat = kafang.main()
        tree = ClassificationTree("entropy")
        bag = TreeBagger(n_trees=15)
        forest = RandomForest(n_trees=15)

        dataset = genfromtxt('/Users/dan/Desktop/Data/N-train.csv', delimiter=',')[1:]
        #train_list = list()
        #for i in useful_feat:
            #train_list.append([x[i] for x in dataset])
        #train_pre = np.array(train_list)
        train= np.array(dataset)
        #train = np.array(mat(train_pre).T)
        print len(train)
        #train = Smote(train,N=100)
        #train = train.over_sampling()

        target_list = genfromtxt('/Users/dan/Desktop/Data/target-2.csv',delimiter= ',')[1:]
        target = np.array(target_list)

        test_list = genfromtxt('/Users/dan/Documents/Python/RandomForest-gh-pages/N-test.csv', delimiter=',' )[1:]
        #test = np.array(test_list)
        test_dataset = genfromtxt('/Users/dan/Desktop/Data/N-test.csv', delimiter=',' )[1:]
        #test_list = list()
        #for i in useful_feat:
            #test_list.append([x[i] for x in test_dataset])
        #test_pre = np.array(test_list)
        test= np.array(test_dataset)
        #test = np.array(mat(test_pre).T)
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
        print results

        result_mat[q] = np.array(results)

    results = list()
    for p in range(12):
        p_dict = dict()
        p_result = result_mat[:, p] #第p个测试集的所有结果
        print p_result #numpy.ndarray
        classes, counts = np.unique(p_result, return_counts=True)
        votemax = np.amax(counts)
        max_index = np.argwhere(counts==votemax)
        p_vote = classes[max_index[0][0]]
        results.append(p_vote)
    print results


    with open("/Users/dan/Desktop/Data/test-results.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "Label"])
        for i, predicted_digit in enumerate(results):
            writer.writerow((i + 1, int(predicted_digit)))
