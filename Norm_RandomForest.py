# -*- coding: utf-8 -*-
### python3.5 ###

import math
import numpy as np
import csv
from collections import defaultdict
import operator
import random
from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt


class DecisionTree(object):
    classIndex = 0
    trainingSamples = []

    def __init__(self, classIndex, trainingSamples):
        self.classIndex = classIndex
        self.trainingSamples = trainingSamples

    def classEntropy(self, data):
        targetClassValuesCount = {} #variable for saving different class labels count
        classEntropy = 0 #variable to save dataset entropy

        for row in data:
            if row[self.classIndex] in targetClassValuesCount:
                targetClassValuesCount[row[self.classIndex]] += 1
            else:
                targetClassValuesCount[row[self.classIndex]] = 1


        for eachKey in targetClassValuesCount:

            classEntropy -= (targetClassValuesCount[eachKey] / len(data)) * math.log(targetClassValuesCount[eachKey] / len(data), 2)
        return classEntropy #Returning calculated entropy

    def getInfoGainForSingleAttribute(self, data, attributeIndex):
        targetAttributeValuesCount = {} #variable for saving different attribute labels count
        attributeEntropy = 0 #variable to save attribute entropy
        classEntropy = self.classEntropy(data) #dataset entropy

        """Logic to find the attribute's different labels count of the dataset provided"""
        for row in data:
            if row[attributeIndex] in targetAttributeValuesCount:
                if row[self.classIndex] in targetAttributeValuesCount[row[attributeIndex]]:
                    targetAttributeValuesCount[row[attributeIndex]][row[self.classIndex]] += 1
                else:
                    targetAttributeValuesCount[row[attributeIndex]][row[self.classIndex]] = 1
            else:
                targetAttributeValuesCount[row[attributeIndex]] = {}
                targetAttributeValuesCount[row[attributeIndex]][row[self.classIndex]] = 1


        entropies = {}
        entropiesTotal = {}
        for eachKey in targetAttributeValuesCount:
            keytotal = 0
            entropies[eachKey] = 0
            for value in targetAttributeValuesCount[eachKey]:
                keytotal += targetAttributeValuesCount[eachKey][value]
            for value in targetAttributeValuesCount[eachKey]:
                entropies[eachKey] -= (targetAttributeValuesCount[eachKey][value] / keytotal) * math.log(targetAttributeValuesCount[eachKey][value] / keytotal, 2)
            entropiesTotal[eachKey] = keytotal
        for eachEntropy in entropies:
            attributeEntropy += ((entropiesTotal[eachEntropy] / len(data)) * entropies[eachEntropy])

        """ Returning InfoGain of the attribute"""
        return classEntropy - attributeEntropy


    def getHighestInfoGainForAttributesRange(self, data, attributesRange):
        allAttributesInfoGain = {} #variable to save infoGain for each attribute provided

        """Getting infoGain for each attribute by calling function getInfoGainForSingleAttribute()"""
        for i in attributesRange:
            allAttributesInfoGain[i] = self.getInfoGainForSingleAttribute(data, i)

        """Finding the attribute with highest InfoGain and returning it's index"""
        allAttributesInfoGain = sorted(allAttributesInfoGain.items(), key=operator.itemgetter(1))
        return (allAttributesInfoGain[len(allAttributesInfoGain)-1][0])   #返回的是节点属性


    def buildDecisionTreeModel(self, data, attributesRange=None):
        """
        This condition is true only first time when data set is provided, so
        for the first time we consider all attributes index except target class
        index.
        """
        if attributesRange is None:
            attributesRange = [i for i in range(0, len(data[0])) if i != self.classIndex]


        """

        target = genfromtxt('target-3.csv', delimiter=',')
        for instance in target:
            if instance in targetClassLabels:
                targetClassLabels[instance] += 1
            else:
                targetClassLabels[instance] = 1
        print (targetClassLabels)
        """
        targetClassLabels = {}
        for instance in data:
            if instance[self.classIndex] in targetClassLabels:
                targetClassLabels[instance[self.classIndex]] += 1
            else:
                targetClassLabels[instance[self.classIndex]] = 1

        targetClassLabels = sorted(targetClassLabels.items(), key=operator.itemgetter(1))
        majorityClassLabel =  targetClassLabels[len(targetClassLabels)-1][0]
        #print (majorityClassLabel)

        """If there is no attribute (as explained above) I'm returning majority class label"""
        if len(attributesRange) == 0:
            return majorityClassLabel

        """If all instances belong to same target class, returning the majority class label"""
        if len(targetClassLabels) == 1:
            return majorityClassLabel

        attributeWithHighestInfoGain = self.getHighestInfoGainForAttributesRange(data, attributesRange)
        decisionTree = {attributeWithHighestInfoGain : {}}

        remainingAttributesRange = [i for i in attributesRange if i != attributeWithHighestInfoGain]

        if len(remainingAttributesRange) != 0:
            random.shuffle(remainingAttributesRange)
            remainingAttributesRange = remainingAttributesRange[:round(len(remainingAttributesRange) * 3 / 4)]


        partitionOfDataForTreesNextLevelTraining = defaultdict(list)   #<class 'collections.defaultdict'>
        for eachInstance in data:
            partitionOfDataForTreesNextLevelTraining[eachInstance[attributeWithHighestInfoGain]].append(eachInstance)

        for eachDataSet in partitionOfDataForTreesNextLevelTraining:
            generateSubTree = self.buildDecisionTreeModel(partitionOfDataForTreesNextLevelTraining[eachDataSet], remainingAttributesRange)
            decisionTree[attributeWithHighestInfoGain][eachDataSet] = generateSubTree

        return decisionTree



    def classifyInstance(self, model, instance, defaultTargetClass):
        if not model: #if the model is empty then returning the majority class label
            return defaultTargetClass
        if not isinstance(model, dict):  # if the node is a leaf, return its class label
            return model
        attributeIndex = list(model.keys())[0]  # using list(dict.keys())
        attributeValues = list(model.values())[0]
        instance_attribute_value = instance[attributeIndex]
        if instance_attribute_value not in attributeValues:  # this value was not in training data
            distance_dict={}
            min_distance = {}
            for key in attributeValues.keys():
                distance = abs(float(instance_attribute_value) - float(key))
                distance_dict[key] = distance
            min_distance = sorted(distance_dict.items(),key=operator.itemgetter(1))
            a = min_distance[0][0]
            return self.classifyInstance(attributeValues[a], instance, defaultTargetClass)
        return self.classifyInstance(attributeValues[instance_attribute_value], instance, defaultTargetClass)


class RandomForest(object):

    nuOfTrees = 0
    treeModels = {}
    trainingSamples = []
    DecisionTreeObjects = {}
    classIndex = 0


    def __init__(self, nuOfTrees, trainingSamples, classIndex):
        self.nuOfTrees = nuOfTrees
        self.trainingSamples = trainingSamples   # type(self.trainingSamples): <class 'numpy.ndarray'> len:38
        self.classIndex = classIndex


    def getRandomBootstrapSamplesWithReplacement(self, sampleSize):
        trainingSampleRange = [x for x in range(len(self.trainingSamples))]
        random.shuffle(trainingSampleRange)
        randomBootstrapSample = []
        for i in trainingSampleRange[:sampleSize]:
            randomBootstrapSample.append(self.trainingSamples[i])    #选28个作为新的训练集
        randomBootstrapSample = np.array(randomBootstrapSample)  #<class 'numpy.ndarray'>

        """Returning the bootstrap sample data with replacement"""
        return randomBootstrapSample


    def getSimpleMajorityVoting(self, instance, defaultLabel):
        predictedClasses = {}  #variable to save predicted value of each model
        """Logic to count the predicted class labels count"""
        for i in range(self.nuOfTrees):
            predictedClass = self.DecisionTreeObjects[i].classifyInstance(self.treeModels[i], instance, defaultLabel)
            if predictedClass in predictedClasses:
                predictedClasses[predictedClass] += 1
            else:
                predictedClasses[predictedClass] = 1

        predictedClasses = sorted(predictedClasses.items(), key=operator.itemgetter(1))
        return predictedClasses[len(predictedClasses)-1][0]


    def buildRandomForest(self):
        bootstrapDataSampleSize = int(round(len(self.trainingSamples) * 3 / 4))

        for i in range(self.nuOfTrees):
            trainingDataForThisTree = self.getRandomBootstrapSamplesWithReplacement(bootstrapDataSampleSize)
            self.DecisionTreeObjects[i] = DecisionTree(self.classIndex, trainingDataForThisTree)
            self.treeModels[i] = self.DecisionTreeObjects[i].buildDecisionTreeModel(trainingDataForThisTree)
            #print (self.treeModels[i])


    def classifyTestData(self, testInstances):
        """Variables to find accuracy"""
        total = 0
        correct = 0

        j=0
        w=0
        for instance in testInstances:
            total += 1
            label = instance[self.classIndex]  #实际的标签
            predicted_label = self.getSimpleMajorityVoting(instance, label)   #预测的标签
            predicted_label = self.getSimpleMajorityVoting(instance,label)
            if label != predicted_label:
                j +=1
            w +=1
            #label = predicted_label
            print (predicted_label)
        print (j)
        accuracy = round(((w-j) * 100) / w, 2), "%"
        print (accuracy)


class RandomForestClassifier(object):

    trainingData = []
    testData = []
    classIndex = 0
    filename = ""
    nuOfTrees = 0

    def __init__(self, filename, nuOfTrees):
        self.nuOfTrees = nuOfTrees
        self.filename = filename

    def createTrainingAndTestSamplesFromData(self):
        rows = csv.reader(open(self.filename))
        self.classIndex = len(next(rows))-1
        data = []
        target = []
        for row in rows:
            data.append(row[:1449])
            target.append(row[self.classIndex])
        #random.shuffle(data)

        data = MinMaxScaler().fit_transform(data)
        data = data.tolist()

        i=0
        for row in data:
            row.append(target[i])
            row = row
            i +=1
        data = np.array(data)

        cutOff = int(round(len(data) * 39/ 51))
        self.trainingData = data[:cutOff]
        self.testData = data[cutOff:]


    def performClassification(self):
        randomForestClassifier = RandomForest(numberOfTrees, self.trainingData, self.classIndex)
        randomForestClassifier.buildRandomForest()
        randomForestClassifier.classifyTestData(self.testData)

numberOfTrees = 10

print("+++++++++++++++++Classification Results for Test Data+++++++++++++++")
randomForestClassifier = RandomForestClassifier("test3.csv", numberOfTrees)
randomForestClassifier.createTrainingAndTestSamplesFromData()
randomForestClassifier.performClassification()


