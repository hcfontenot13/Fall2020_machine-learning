# Lab 3

#  Logistic regression, Naive Bayes, Generative vs Discriminative models

import numpy as np
import torch
from torch.nn import functional as F

from matplotlib import pyplot as plt
from sklearn import datasets


# Load the Iris dataset
iris = datasets.load_iris()
print(iris.DESCR)

X = iris.data
Y = iris.target
Y = Y.reshape(-1, 1)
nclass = len(np.unique(Y))


# Examine feature correlations
print(np.corrcoef(X[:, 0], Y[:, 0]))
print(np.corrcoef(X[:, 1], Y[:, 0]))
print(np.corrcoef(X[:, 2], Y[:, 0]))  # strong correlation
print(np.corrcoef(X[:, 3], Y[:, 0]))  # strong correlation

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(X[:, 0], Y, 'o')
plt.xlabel('x1')
plt.ylabel('Y')
plt.subplot(2, 2, 2)
plt.plot(X[:, 1], Y, 'o')
plt.xlabel('x2')
plt.ylabel('Y')
plt.subplot(2, 2, 3)
plt.plot(X[:, 2], Y, 'o')
plt.xlabel('x3')
plt.ylabel('Y')
plt.subplot(2, 2, 4)
plt.plot(X[:, 3], Y, 'o')
plt.xlabel('x4')
plt.ylabel('Y')
plt.show()

plt.figure()
plt.plot(X[:, 2][Y[:, 0] == 0], X[:, 3][Y[:, 0] == 0], 'o')
plt.plot(X[:, 2][Y[:, 0] == 1], X[:, 3][Y[:, 0] == 1], 'x')
plt.plot(X[:, 2][Y[:, 0] == 2], X[:, 3][Y[:, 0] == 2], '^')
plt.xlabel('x3')
plt.ylabel('x4')
plt.legend(['Class 0', 'Class 1', 'Class 2'])
plt.show()


# Train/test split: ~85/15
idx = np.random.permutation(X.shape[0])
trn_idx, tst_idx = idx[:130], idx[130:]
X_train, X_test = X[trn_idx, :], X[tst_idx, :]
Y_train, Y_test = Y[trn_idx, :], Y[tst_idx, :]


# Logistic regression (discriminative)
class LogisticRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def train(X_train, Y_train, epochs):
    model = LogisticRegression(X_train.shape[1], nclass)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels.squeeze().long())

        loss.backward()
        optimizer.step()

        print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model


def classification_results(y_pred, y_actual):
    print(np.vstack((y_pred, y_actual)))
    return sum(y_pred == y_actual) / len(y_pred)


# Train using all features
model1 = train(X_train=X_train, Y_train=Y_train, epochs=100)
with torch.no_grad():
    y_predicted = model1(torch.from_numpy(X_test).float())
_, pred = torch.max(y_predicted, 1)
print('Accuracy: ', classification_results(pred.numpy(), Y_test[:, 0]))


# Use only the two features with strong correlation
model2 = train(X_train=X_train[:, 2:4], Y_train=Y_train, epochs=20)
with torch.no_grad():
    y_predicted = model2(torch.from_numpy(X_test[:, 2:4]).float())
_, pred = torch.max(y_predicted, 1)
print('Accuracy: ', classification_results(pred.numpy(), Y_test[:, 0]))


# Define with softmax activation
"""
NOTE:Softmax is the generalization of the sigmoid function. 
So if you are doing binary classification you may use sigmoid (like in the commented out line 125).
But for multiclass classification you need to use softmax.
"""


class LogisticRegression1(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LogisticRegression1, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.transform = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # out = F.sigmoid(self.linear(x))
        out = self.transform(self.linear(x))
        return out


def train1(X_train, Y_train, epochs):
    model = LogisticRegression1(X_train.shape[1], nclass)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels.squeeze().long())

        loss.backward()
        optimizer.step()

        print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model


# Test it out
model3 = train1(X_train=X_train, Y_train=Y_train, epochs=100)
with torch.no_grad():
    y_predicted = model3(torch.from_numpy(X_test).float())
_, pred = torch.max(y_predicted, 1)
print('Accuracy: ', classification_results(pred.numpy(), Y_test[:, 0]))


# Naive Bayes (generative)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train[:, 0]).predict(X_test)
print('Accuracy: ', classification_results(y_pred, Y_test[:, 0]))


# Cross-validation
K = 5
splits = np.split(idx, K)
acc_all_lr = np.zeros((K, 1))
acc_all_nb = np.zeros((K, 1))

for k in range(K):
    tst_idx = splits[k]
    trn_idx = np.delete(idx, splits[k])
    X_train, X_test = X[trn_idx, :], X[tst_idx, :]
    Y_train, Y_test = Y[trn_idx], Y[tst_idx]

    modelK = train(X_train=X_train, Y_train=Y_train, epochs=100)
    with torch.no_grad():
        y_predicted = modelK(torch.from_numpy(X_test).float())
    _, pred = torch.max(y_predicted, 1)

    acc_all_lr[k] = classification_results(pred.numpy(), Y_test[:, 0])


for k in range(K):
    tst_idx = splits[k]
    trn_idx = np.delete(idx, splits[k])
    X_train, X_test = X[trn_idx, :], X[tst_idx, :]
    Y_train, Y_test = Y[trn_idx], Y[tst_idx]

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, Y_train[:, 0]).predict(X_test)

    acc_all_nb[k] = classification_results(pred.numpy(), Y_test[:, 0])


# Display cross-validated results for both learners
print('Cross-Validated Accuracy: Log.Regression: ', acc_all_lr.mean())
print('Cross-Validated Accuracy: Naive Bayes: ', acc_all_nb.mean())
