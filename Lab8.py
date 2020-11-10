# Lab 8: SVM
# 11/10/2020

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets, svm, metrics

import numpy as np
from math import floor, ceil

from matplotlib import pyplot as plt
import seaborn as sns


# SVM: Classification
cancer = datasets.load_breast_cancer()
print(cancer.DESCR)

data = cancer.data
target = cancer.target

print(data.shape)

# Split into test/train
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
print(X_train.shape)
print(X_test.shape)

# Train/test SVM classifier
model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Error metrics
print('Accuracy: {}'.format(metrics.accuracy_score(y_test, y_pred)))
print(metrics.confusion_matrix(y_test, y_pred))

# Plots
ax = sns.heatmap(metrics.confusion_matrix(y_test, y_pred), cbar=False, annot=True, annot_kws={'fontsize': 20})
plt.xlabel('Y_test', fontsize=20)
plt.ylabel('Y_pred', fontsize=20)
ax.set_ylim([0, 2])
ax.invert_yaxis()

# Hyperparameter tuning using grid search
param_grid = {
    'C': [0.1, 1, 10, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
    'kernel': ['rbf']
}

grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)

# Best hyperparameters
print(grid.best_params_)
print(grid.best_estimator_)

y_pred = grid.predict(X_test)
print('Accuracy: {}'.format(metrics.accuracy_score(y_test, y_pred)))
print(metrics.confusion_matrix(y_test, y_pred))

ax = sns.heatmap(metrics.confusion_matrix(y_test, y_pred), cbar=False, annot=True, annot_kws={'fontsize': 20})
plt.xlabel('Y_test', fontsize=20)
plt.ylabel('Y_pred', fontsize=20)
ax.set_ylim([0, 2])
ax.invert_yaxis()


# SVR: Regression
boston = datasets.load_boston()
data = boston.data
target = boston.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
print(X_train.shape)
print(X_test.shape)

# Train SVM for regression
model = svm.SVR(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Error metrics
print('RMSE: {}'.format(np.sqrt(np.mean((y_pred - y_test) ** 2))))
print('MAPE: {}'.format(np.mean(np.abs((y_pred - y_test) / y_test))))

# PLots
x1 = floor(min(y_test))
x2 = ceil(max(y_test))
plt.plot(y_test, y_pred, 'o', label='Prediction')
plt.plot(list(range(x1, x2)), list(range(x1, x2)), label='Perfect prediction')
plt.legend()
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
    'kernel': ['rbf']
}

grid = GridSearchCV(svm.SVR(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)

# Best hyperparameters
print(grid.best_params_)
print(grid.best_estimator_)

y_pred = grid.predict(X_test)
print('RMSE: {}'.format(np.sqrt(np.mean((y_pred - y_test) ** 2))))
print('MAPE: {}'.format(np.mean(np.abs((y_pred - y_test) / y_test))))

plt.plot(y_test, y_pred, 'o', label='Prediction')
plt.plot(list(range(x1, x2)), list(range(x1, x2)), label='Perfect prediction')
plt.legend()
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
