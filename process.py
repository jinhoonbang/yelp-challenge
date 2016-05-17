import pandas as pd
import numpy as np
import os.path
import operator
import glob
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

curr_dir = os.path.dirname(os.path.realpath(__file__))
data = np.fromfile(os.path.join(curr_dir, "raw_data.bin"), dtype='float64')
n_row = int(data[0])
n_col= int(data[1])
data = np.delete(data, [0, 1])
data = data.reshape((n_row, n_col))

print("data shape: {}", data.shape)

label = data[:,0]
feature = data[:,1:]

print("feature.shape: {}", feature.shape)

pca = PCA(n_components=1000
feature = pca.fit_transform(feature)

print("##PCA##")
print("feature.shape: {}", feature.shape)

count = data.shape[0]
train_split = int(count * 0.6)
test_split = int(count * 0.8)

x_train = feature[:train_split]
x_valid = feature[train_split:test_split]
x_test = feature[test_split:]
y_train = label[:train_split]
y_valid = label[train_split:test_split]
y_test = label[test_split:]

print("label.shape: {}", label.shape)
label = np.reshape(label, (n_row, 1))
print("label.shape: {}", label.shape)

data = np.concatenate((label, feature), axis = 1)
dimension = np.array([data.shape[0], data.shape[1]])
data=np.append(dimension,data)
data.astype("float64")
data.tofile(os.path.join(curr_dir, "data_pca_1000.bin"))

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
predicted = rfc.predict(x_test)

print(f1_score(y_test, predicted, average="micro"))
print(f1_score(y_test, predicted, average="macro"))
print(f1_score(y_test, predicted, average="weighted"))
print(classification_report(y_test, predicted))
