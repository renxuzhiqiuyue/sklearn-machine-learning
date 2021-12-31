import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.data import wine_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pdb


if __name__ == '__main__':
    #加载数据, 并且划分数据集
    X, y = wine_data()
    train_x, test_x, train_y, test_y = train_test_split(X, y, 
                                                   stratify=y, test_size=0.3, random_state=1)
    print("Train_x Shape :: ", train_x.shape) 
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    # 利用随机森林分类进行筛选
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)

    predictions = clf.predict(test_x)

    for i in range(0, 5):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

    print("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print(" Confusion matrix ", confusion_matrix(test_y, predictions))