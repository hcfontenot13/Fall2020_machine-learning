# Lab 3

# Logistic Regression, Naive Bayes, Generative vs Discriminative Models

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
print(np.corrcoef(X[:, 2], Y[:, 0]))
print(np.corrcoef(X[:, 3], Y[:, 0]))

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


# Train/test split: ~80/20
idx = np.random.permutation(X.shape[0])
trn_idx, tst_idx = idx[:142], idx[142:]
X_train, X_test = X[trn_idx, :], X[tst_idx, :]
Y_train, Y_test = Y[trn_idx], Y[tst_idx]


# Logistic regression (note similarity to linear regression!)
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
    print(np.vstack((y_pred, y_actual, ['T' if y_pred[i] == y_actual[i] else 'F' for i in range(len(y_pred))])))
    return sum(y_pred == y_actual) / len(y_pred)


# Train using all features
model_logr = train(X_train=X_train, Y_train=Y_train, epochs=20)
with torch.no_grad():
    y_predicted = model_logr(torch.from_numpy(X_test).float())
_, pred = torch.max(y_predicted, 1)
print(classification_results(pred.numpy(), Y_test[:, 0]))

# Use only the two features most highly correlated with target
model_1 = train(X_train=X_train[:, 2:4], Y_train=Y_train, epochs=20)
with torch.no_grad():
    y_predicted = model_1(torch.from_numpy(X_test[:, 2:4]).float())
_, pred = torch.max(y_predicted, 1)
print(classification_results(pred.numpy(), Y_test[:, 0]))


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train[:, 0]).predict(X_test)
print(classification_results(y_pred, Y_test[:, 0]))


# Cross-validation: 5-fold

K = 5
splits = np.split(idx, K)
acc_all = np.zeros((K, 1))
for k in range(K):
    tst_idx = splits[k]
    trn_idx = np.delete(idx, splits[k])
    X_train, X_test = X[trn_idx, :], X[tst_idx, :]
    Y_train, Y_test = Y[trn_idx], Y[tst_idx]

    modelK = train(X_train=X_train, Y_train=Y_train, epochs=20)
    with torch.no_grad():
        y_predicted = modelK(torch.from_numpy(X_test).float())
    _, pred = torch.max(y_predicted, 1)

    acc_all[k] = classification_results(pred.numpy(), Y_test[:, 0])

print('Cross-validated Accuracy: ', acc_all.mean())

for k in range(K):
    tst_idx = splits[k]
    trn_idx = np.delete(idx, splits[k])
    X_train, X_test = X[trn_idx, :], X[tst_idx, :]
    Y_train, Y_test = Y[trn_idx], Y[tst_idx]

    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, Y_train[:, 0]).predict(X_test)

    acc_all[k] = classification_results(y_pred, Y_test[:, 0])

print('Cross-validated Accuracy: ', acc_all.mean())

