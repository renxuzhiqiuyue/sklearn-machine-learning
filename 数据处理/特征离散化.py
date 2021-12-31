#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
ages = pd.DataFrame({'years':[10, 14, 30, 53, 67, 32, 45], 'name':['A', 'B', 'C', 'D', 'E', 'F', 'G']})
ages_cut = pd.cut(ages['years'],3)
#print(ages_cut)

ages_cut = pd.qcut(ages['years'],3)
#print(ages_cut)

ages2 = pd.DataFrame({'years':[10, 14, 30, 53, 300, 32, 45], 'name':['A', 'B', 'C', 'D', 'E', 'F', 'G']})
klass2 = pd.cut(ages2['years'], 3, labels=['Young', 'Middle', 'Senior'])    # ②
ages2['label'] = klass2
#print(ages2)

from sklearn.preprocessing import KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')   # ④
trans = kbd.fit_transform(ages[['years']])    # ⑤
print(trans)
ages['kbd'] = trans[:, 0]    # ⑥
print(ages)

# 有监督离散化
