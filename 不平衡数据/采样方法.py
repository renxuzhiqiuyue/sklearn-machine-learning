from sklearn import datasets
import numpy as np
iris = datasets.load_iris() # 导入数据集
X = iris.data # 获得其特征向量
y = iris.target # 获得样本label

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)
print('上采样：', X_resampled, y_resampled)

from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler(replacement=True)
X_resampled, y_resampled = ros.fit_resample(X, y)
print('下采样：', X_resampled, y_resampled)

from imblearn.over_sampling import SMOTE
X_resampled_smote, y_resampled_smote = SMOTE().fit_resample(X, y)
print('SMOTE：', X_resampled, y_resampled)

from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, y)
print('Borderline-SMOTE：', X_resampled, y_resampled)

from imblearn.over_sampling import ADASYN
X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_resample(X, y)
print('ADASYN：', X_resampled_adasyn, y_resampled_adasyn)

from sklearn.svm import SVC #SVM中的分类算法SVC
model_svm = SVC(class_weight='balanced') # 创建SVC模型对象并指定类别权重
model_svm.fit(X, y) # 输入x和y并训练模型