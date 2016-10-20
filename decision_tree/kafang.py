#coding: utf-8
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

def main():
    data_folder = os.path.join(os.path.expanduser("/Users/dan/Desktop/"), "Data")
    data_filename = os.path.join(data_folder, "N-train.csv")
    target_filename = os.path.join(data_folder, "target-2.csv")

    data = pd.read_csv(data_filename)
    target = pd.read_csv(target_filename)
    data.dropna(how='all', inplace=True)

    #vt = VarianceThreshold()    #VarianceThreshold转换器可用来删除特征值的方差达不到最低标准的特征
    X = data.values
    X = MinMaxScaler().fit_transform(X)        #  归一化
    #Xt = vt.fit_transform(X)
    y = (target["class"] == 1 ).values
    transformer = SelectKBest(score_func=chi2, k=3)
    Xt_chi2 = transformer.fit_transform(X, y)
    #print (transformer.scores_)
    sorted_feat = np.argsort(-transformer.scores_)
    useful_feat = sorted_feat[0:100]
    print useful_feat
    return useful_feat

if __name__ == '__main__':
    main()
