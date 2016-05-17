import pandas as pd
import numpy as np
import os.path
import operator
import glob
import pickle
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score

filename = "data_pca_1000.bin"

curr_dir = os.path.dirname(os.path.realpath(__file__))
data = np.fromfile(os.path.join(curr_dir, filename), dtype='float64')
n_row = int(data[0])
n_col= int(data[1])
data = np.delete(data, [0, 1])
data = data.reshape((n_row, n_col))

print("data shape: {}", data.shape)

label = data[:,0]
feature = data[:,1:]

print("feature.shape: {}", feature.shape)

count = data.shape[0]
split = int(count * 0.7)

x_train = feature[:split]
x_test = feature[split:]
y_train = label[:split]
y_test = label[split:]

parameter = {'C':[1, 0.1, 0.01], 'penalty':('l1','l2')}
gs = GridSearchCV(LogisticRegression(multi_class = "ovr"), parameter, cv=5,
                       scoring='f1_weighted')

gs.fit(x_train, y_train)
predicted = gs.predict(x_test)

print("###Best Estimator")
print("best_estimator {}", gs.best_estimator_)
print("grid_scores_ {}", gs.grid_scores_)
print("best_score {}", gs.best_score_)
print("best_params {}", gs.best_params_)

print("####Accuracy")
print("accuracy {}".format(accuracy_score(y_test, predicted)))

print("####F1 Score")
print("micro {}".format(f1_score(y_test, predicted, average="micro")))
print("macro {}".format(f1_score(y_test, predicted, average="macro")))
print("weighted {}".format(f1_score(y_test, predicted, average="weighted")))
print(classification_report(y_test, predicted))
