from sklearn import preprocessing
import numpy as np

X = np.array([[1, -1, 2],
              [2, 0, 0],
              [0, 1, -1]])
# X_scaled = preprocessing.scale(X)
# print(X_scaled)
# print(X_scaled.mean(axis=0))
# print(X_scaled.std(axis=0))

scaler = preprocessing.StandardScaler().fit(X)
print(scaler.mean_)
print(scaler.scale_)
print(scaler)