#@Time      :2018/12/3 20:44
#@Author    :zhounan
# @FileName: demo.py
import numpy as np
from classfier.label_powerset import LabelPowerset
from sklearn.ensemble import RandomForestClassifier

def load_data(dataset_name):
    x_train = np.load('./dataset/' + dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/' + dataset_name + '/y_train.npy')
    x_test = np.load('./dataset/' + dataset_name + '/x_test.npy')
    y_test = np.load('./dataset/' + dataset_name + '/y_test.npy')

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    dataset_names = ['yeast', 'delicious']
    dataset_name = dataset_names[0]
    x_train, y_train, x_test, x_test = load_data(dataset_name)

    classfier = LabelPowerset(classfier=RandomForestClassifier(n_estimators=100))
    classfier.fit(x_train, y_train)
    predictions = classfier.predict(x_test)
