# Lab 5: Hyperparameter tuning

import numpy as np
import torch

# Define a network
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# Training function
def train(X_train, Y_train, H, learning_rate, epochs=500):
    model = TwoLayerNet(X_train.shape[1], H, Y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

    return model


# K-fold cross-validation
from sklearn.model_selection import KFold
def kfold_CV(X, Y, K, H, learning_rate):
    hidden = H
    lr = learning_rate

    kf = KFold(n_splits=K, shuffle=False)
    rmse_trn_cv, rmse_tst_cv = np.empty(0), np.empty(0)

    for trn_idx, tst_idx in kf.split(X):
        X_train, X_test = X[trn_idx, :], X[tst_idx, :]
        Y_train, Y_test = Y[trn_idx], Y[tst_idx]

        modelK = train(X_train=X_train, Y_train=Y_train, H=hidden, learning_rate=lr)
        with torch.no_grad():
            yhat_trn = modelK(torch.from_numpy(X_train).float()).numpy()
            yhat_tst = modelK(torch.from_numpy(X_test).float()).numpy()

        rmse_trn = np.sqrt(np.mean((yhat_trn - Y_train) ** 2))
        rmse_tst = np.sqrt(np.mean((yhat_tst - Y_test) ** 2))

        rmse_trn_cv = np.append(rmse_trn_cv, rmse_trn)
        rmse_tst_cv = np.append(rmse_tst_cv, rmse_tst)

    return rmse_trn_cv.mean(), rmse_tst_cv.mean()


# Dataset
N, D_in, D_out = 500, 5, 1
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)


# Hyperparameter tuning: Grid Search
H_list = list(range(2, 6))
lr_list = [1e-4, 1e-3, 1e-2]

rmse_trn = np.zeros((len(H_list), len(lr_list)))
rmse_tst = np.zeros_like(rmse_trn)

for h, H in enumerate(H_list):
    for l, lr in enumerate(lr_list):
        trn_val, tst_val = kfold_CV(x, y, 5, H, lr)
        rmse_trn[h, l] = trn_val
        rmse_tst[h, l] = tst_val

        print('H = {}, lr = {}: Training RMSE = {}, Testing RMSE = {}'.format(H, lr, trn_val, tst_val))

i, j = np.argwhere(rmse_tst == np.min(rmse_tst))[0]
h_best, lr_best = H_list[i], lr_list[j]


# Overfitting
H_list = list(range(2, 51))
H_rmse_trn, H_rmse_tst = np.empty(0), np.empty(0)

for h, H in enumerate(H_list):
    trn_val, tst_val = kfold_CV(x, y, 5, H, learning_rate=0.001)
    H_rmse_trn = np.append(H_rmse_trn, trn_val)
    H_rmse_tst = np.append(H_rmse_tst, tst_val)

import matplotlib.pyplot as plt
plt.plot(H_list, H_rmse_trn, label='train')
plt.plot(H_list, H_rmse_tst, label='test')
plt.xlabel('Hidden Layer Neurons')
plt.ylabel('Cross-validated RMSE')
plt.legend()
