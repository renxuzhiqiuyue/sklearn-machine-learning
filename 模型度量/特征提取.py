#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 添加其他环境的包
import sys
sys.path.append('D:\\Anaconda3\\Lib\\site-packages')
sys.path.append('D:\\Anaconda3\\envs\\pytorch\\Lib\\site-packages')
sys.path.append('D:\\Anaconda3\\envs\\python37\\Lib\\site-packages')

# 循序法特征提取
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import wine_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = wine_data()
X_train, X_test, y_train, y_test= train_test_split(X, y, 
                                                   stratify=y,
                                                   test_size=0.3,
                                                   random_state=1)
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
knn = KNeighborsClassifier(n_neighbors=3)    # ①
sfs = SFS(estimator=knn,     # ②
           k_features=4,
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=0)
sfs.fit(X_train_std, y_train)

# 画图
knn = KNeighborsClassifier(n_neighbors=3)
sfs2 = SFS(estimator=knn,     # ⑤
           k_features=(3, 10),
           forward=True, 
           floating=True,   
           verbose=0,
           scoring='accuracy',
           cv=5)
sfs2.fit(X_train_std, y_train)

#%matplotlib inline
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig = plot_sfs(sfs2.get_metric_dict(), kind='std_err')