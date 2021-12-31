#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
g = pd.DataFrame({"gender": ["man", 'woman', 'woman', 'man', 'woman']})
pd.get_dummies(g) # g可以单列也可以多列
pd.get_dummies(g['gender'], drop_first=True) #抛弃第一列

#进行处理后，可以改为零和约束
g_cut = pd.get_dummies(g) # g可以单列也可以多列
g_cut[g_cut.columns[0]] = -1
print(g_cut)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
features = ohe.fit_transform(g[['gender']]) #请注意这里
print(features.toarray())
#print(features)

