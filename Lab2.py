"""Lab 2"""
"""In this lab:

PyTorch introductory tutorial
-Creating a toy dataset
-Plotting distributions/correlations between variables
-Split into train/test sets
-Linear regression
-Ridge regression
-Lasso
-Reporting error metrics: RMSE, MAPE, MAE, MBE, R2
-Cross-validation

"""

import numpy as np
from matplotlib import pyplot as plt
import torch


# Create a toy dataset: Y = 3*x1 + 0.5*x2 - 2*x3 + 6
np.random.seed(3)
X = np.random.randn(1000, 3)
Y = 3 * X[:, 0] + 0.5 * X[:, 1] - 2 * X[:, 2] + 6.0
Y += np.random.randn(1000)
Y = Y.reshape(-1, 1)


# Find some relationships between variables
print(np.corrcoef(X[:, 0], Y[:, 0]))
print(np.corrcoef(X[:, 1], Y[:, 0]))
print(np.corrcoef(X[:, 2], Y[:, 0]))

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
plt.hist(Y[:, 0], bins=30)
plt.title('Histogram of Y')


# Train/test split: 80/20
idx = np.random.permutation(X.shape[0])
trn_idx, tst_idx = idx[:800], idx[800:]
X_train, X_test = X[trn_idx, :], X[tst_idx, :]
Y_train, Y_test = Y[trn_idx], Y[tst_idx]


# Linear regression
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def train(X_train, Y_train, epochs, ridge=None, lasso=None):
    model = linearRegression(X_train.shape[1], Y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if ridge:
        optimizer.param_groups[0]['weight_decay'] = 5

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        if lasso:
            l1_norm = torch.norm(model.linear.weight, p=1)
        else:
            l1_norm = 0
        loss = criterion(outputs, labels)
        loss += l1_norm
        loss.backward()

        optimizer.step()
        print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model


model1 = train(X_train=X_train, Y_train=Y_train, epochs=500)
with torch.no_grad():
    y_predicted = model1(torch.from_numpy(X_test).float())

print(model1.linear.weight.detach())
print(model1.linear.bias.detach())


def plot_results(Y_test, y_predicted):
    figx = plt.figure()
    plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
    plt.plot(range(len(Y_test)), y_predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    return figx


fig_LR = plot_results(Y_test, y_predicted)


plt.figure()
plt.plot(X[:, 0], Y[:, 0], 'o')
plt.plot(X[:, 0], model1.linear.weight.detach()[0][0].numpy() * X[:, 0] + model1.linear.bias.detach().numpy())

plt.figure()
plt.plot(X[:, 1], Y[:, 0], 'o')
plt.plot(X[:, 1], model1.linear.weight.detach()[0][1].numpy() * X[:, 1] + model1.linear.bias.detach().numpy())

plt.figure()
plt.plot(X[:, 2], Y[:, 0], 'o')
plt.plot(X[:, 2], model1.linear.weight.detach()[0][2].numpy() * X[:, 2] + model1.linear.bias.detach().numpy())


def error_metrics(Y_test, y_pred):
    # RMSE, MAPE, MAE, MBE, R2
    rmse = np.sqrt(np.mean((y_pred - Y_test) ** 2))
    mape = np.mean(np.abs(y_pred - Y_test) / Y_test)
    mae = np.mean(np.abs(y_pred - Y_test))
    mbe = np.mean(y_pred - Y_test)
    r2 = np.corrcoef(y_pred.squeeze(), Y_test.squeeze())[0, 1]**2
    return rmse, mape, mae, mbe, r2


rmse1, mape1, mae1, mbe1, r21 = error_metrics(Y_test, y_predicted.numpy())


# Ridge regression
model2 = train(X_train=X_train, Y_train=Y_train, epochs=500, ridge=True)
with torch.no_grad():
    y_predicted = model2(torch.from_numpy(X_test).float())

print(model2.linear.weight.detach())
print(model2.linear.bias.detach())

fig_ridge = plot_results(Y_test, y_predicted)

plt.figure()
plt.plot(X[:, 0], Y[:, 0], 'o')
plt.plot(X[:, 0], model2.linear.weight.detach()[0][0].numpy() * X[:, 0] + model2.linear.bias.detach().numpy())

plt.figure()
plt.plot(X[:, 1], Y[:, 0], 'o')
plt.plot(X[:, 1], model2.linear.weight.detach()[0][1].numpy() * X[:, 1] + model2.linear.bias.detach().numpy())

plt.figure()
plt.plot(X[:, 2], Y[:, 0], 'o')
plt.plot(X[:, 2], model2.linear.weight.detach()[0][2].numpy() * X[:, 2] + model2.linear.bias.detach().numpy())

rmse2, mape2, mae2, mbe2, r22 = error_metrics(Y_test, y_predicted.numpy())


# Lasso
model3 = train(X_train=X_train, Y_train=Y_train, epochs=500, lasso=True)
with torch.no_grad():
    y_predicted = model3(torch.from_numpy(X_test).float())

print(model3.linear.weight.detach())
print(model3.linear.bias.detach())

fig_lasso = plot_results(Y_test, y_predicted)

y_pred = y_predicted.numpy()

rmse3, mape3, mae3, mbe3, r23 = error_metrics(Y_test, y_predicted.numpy())


# Cross-validation: 5-fold
K = 5
splits = np.split(idx, K)
rmse_all = np.zeros((K, 1))
for k in range(K):
    tst_idx = splits[k]
    trn_idx = np.delete(idx, splits[k])
    X_train, X_test = X[trn_idx, :], X[tst_idx, :]
    Y_train, Y_test = Y[trn_idx], Y[tst_idx]

    modelK = train(X_train=X_train, Y_train=Y_train, epochs=500)
    with torch.no_grad():
        y_predicted = modelK(torch.from_numpy(X_test).float())

    rmse, _, _, _, _ = error_metrics(Y_test, y_predicted.numpy())
    rmse_all[k] = rmse

print('Cross-validated RMSE: ', rmse_all.mean())
