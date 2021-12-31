import lightgbm as lgb
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#划分数据集
file = datasets.load_boston()
data = file.data
target = file.target
X_train, X_test, y_train, y_test = train_test_split(
        data,target,test_size = 0.4,random_state = 0)

lgb_train = lgb.Dataset(X_train,y_train)
lgb_eval = lgb.Dataset(X_test,y_test,reference=lgb_train)

params = {'task':'train',
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric':{'l2','mae'},
    'num_leaves':31,
    'learning_rate':0.05,
    'feature_fraction':0.9,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'verbose':0}

#使用选择的参数进行训练

gbm = lgb.train(params,lgb_train,
    num_boost_round=100,
    valid_sets=lgb_eval,
    early_stopping_rounds=5, 
    verbose_eval=False)  
#verbose_eval=False这个参数是控制每次训练不输出迭代效果参数

#下面进行预测
lgb_predit = gbm.predict(X_test,num_iteration=gbm.best_iteration) 

#输出均方根误差
print(mean_squared_error(y_test,lgb_predit) ** 0.5)     


### 保存模型特征重要性，方便后面特征选择
df = pd.DataFrame(['feature '+str(i) for i in range(len(data[0]))], columns=['feature'])
df['importance']=list(gbm.feature_importance())                
# 特征分数

df = df.sort_values(by='importance',ascending=False)                      
# 特征按照得分排序

df.to_csv("e:/data/sklearn/feature_score.csv",index=None,encoding='gbk')