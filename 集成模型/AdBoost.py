# encoding=utf-8

import pandas as pd
import time

from sklearn.model_selection import train_test_split
from mlxtend.data import wine_data
from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
    print("Start read data...")
    time_1 = time.time()
    # 随机选取33%数据作为测试集，剩余为训练集
    X, y = wine_data()
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, 
                                                   stratify=y, test_size=0.3, random_state=1)
    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training...') 
    # n_estimators表示要组合的弱分类器个数；
    # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
    clf = AdaBoostClassifier(n_estimators=100,algorithm='SAMME.R')
    clf.fit(train_features,train_labels)
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))


    print('Start predicting...')
    test_predict = clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))


    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)