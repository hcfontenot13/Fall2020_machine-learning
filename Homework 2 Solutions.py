
import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn import datasets

# Load dataset and view description
boston = datasets.load_boston()
print(boston.DESCR)

# Set training and target data
data = boston.data
target = boston.target
names = boston.feature_names.tolist()

target = target.reshape(-1, 1)
x_train, y_train = data[:450, ], target[:450, ]
x_test, y_test = data[:450, ], target[:450, ]

# ------------------------------------------------------------
# Problem 1: Explore relationships between features of the data
# ------------------------------------------------------------

R = []
print('Correlation between features and target:')
for i, feature in enumerate(names):
    r = np.corrcoef(data[:, i], target[:, 0])[0, 1]
    if abs(r) > 0.4:
        R.append(i)
    s = '*' if abs(r) > 0.4 else ''
    print(feature, '\t', r, '\t', s)

# Plot the features with correlation > 0.4
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.plot(data[:, R[i]], target, 'o')
    plt.xlabel(names[R[i]])
    plt.ylabel('MEDV')
    plt.title('r = {}'.format(np.corrcoef(data[:, R[i]], target[:, 0])[0, 1]))
plt.show()

# Histogram of target labels
plt.figure()
plt.hist(target, bins=50)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('MEDV')


# ------------------------------------------------------------
# Define LinearRegression class, train function, error function
# ------------------------------------------------------------

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def train(x_train, y_train, epochs, lr, ridge=None, lasso=None, reg_lambda=None):
    model = LinearRegression(x_train.shape[1], y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ridge:
        optimizer.param_groups[0]['weight_decay'] = reg_lambda

    for epoch in range(epochs):
        inputs = torch.from_numpy(x_train).float()
        labels = torch.from_numpy(y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        if lasso:
            l1_norm = reg_lambda * torch.norm(model.linear.weight, p=1)
        else:
            l1_norm = 0
        loss = criterion(outputs, labels)
        loss += l1_norm
        loss.backward()

        optimizer.step()
        print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model


def error_metrics(y_test, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mape = np.mean(np.abs(y_pred - y_test) / y_test)
    mae = np.mean(np.abs(y_pred - y_test))
    mbe = np.mean(y_pred - y_test)
    r2 = np.corrcoef(y_pred.squeeze(), y_test.squeeze())[0, 1] ** 2
    return rmse, mape, mae, mbe, r2


# ------------------------------------------------------------
# Problem 2: Linear regression
# ------------------------------------------------------------

model_lr = train(x_train=x_train, y_train=y_train, epochs=1000, lr=0.1)
with torch.no_grad():
    y_predicted = model_lr(torch.from_numpy(x_test).float()).numpy()

print('Trained model coefficients: ', model_lr.linear.weight.detach().numpy())
print('Trained model bias: ', model_lr.linear.bias.detach().numpy())

rmse_lr, mape_lr, mae_lr, mbe_lr, r2_lr = error_metrics(y_test, y_predicted)

plt.figure()
plt.plot(y_predicted)
plt.plot(y_test)
plt.legend(['Predicted', 'Measured'])
plt.ylabel('MEDV')
plt.title('Linear Regression (RMSE = {})'.format(rmse_lr))

plt.figure()
plt.plot(np.abs(y_predicted - y_train), 'o')
plt.legend(['absolute error'])

# ------------------------------------------------------------
# Problem 3: Ridge regression
# ------------------------------------------------------------

model_rdg = train(x_train=x_train, y_train=y_train, epochs=1000, lr=0.1, ridge=True, reg_lambda=0.5)
with torch.no_grad():
    y_predicted = model_rdg(torch.from_numpy(x_test).float()).numpy()

print('Trained model coefficients: ', model_rdg.linear.weight.detach().numpy())
print('Trained model bias: ', model_rdg.linear.bias.detach().numpy())

rmse_rdg, mape_rdg, mae_rdg, mbe_rdg, r2_rdg = error_metrics(y_test, y_predicted)

plt.figure()
plt.plot(y_predicted)
plt.plot(y_test)
plt.legend(['Predicted', 'Measured'])
plt.ylabel('MEDV')
plt.title('Ridge Regression (RMSE = {})'.format(rmse_rdg))

plt.figure()
plt.plot(np.abs(y_predicted - y_train), 'o')
plt.legend(['absolute error'])

# ------------------------------------------------------------
# Problem 4: Lasso
# ------------------------------------------------------------

model_lso = train(x_train=x_train, y_train=y_train, epochs=1000, lr=0.1, lasso=True, reg_lambda=5)
with torch.no_grad():
    y_predicted = model_lso(torch.from_numpy(x_test).float()).numpy()

print('Trained model coefficients: ', model_lso.linear.weight.detach().numpy())
print('Trained model bias: ', model_lso.linear.bias.detach().numpy())

rmse_lso, mape_lso, mae_lso, mbe_lso, r2_lso = error_metrics(y_test, y_predicted)

plt.figure()
plt.plot(y_predicted)
plt.plot(y_test)
plt.legend(['Predicted', 'Measured'])
plt.ylabel('MEDV')
plt.title('Lasso (RMSE = {})'.format(rmse_lso))

plt.figure()
plt.plot(np.abs(y_predicted - y_train), 'o')
plt.legend(['absolute error'])

# ------------------------------------------------------------
# Problem 5: Cross-validated Linear Regression
# ------------------------------------------------------------

K = 5
idx = np.array(list(range(x_train.shape[0])))
splits = np.split(idx, K)
rmse_all = np.zeros((K, 1))
mape_all = np.zeros((K, 1))
mae_all = np.zeros((K, 1))
mbe_all = np.zeros((K, 1))
r2_all = np.zeros((K, 1))
for k in range(K):
    tst_idx = splits[k]
    trn_idx = np.delete(idx, splits[k])
    x_trn, x_tst = data[trn_idx, :], data[tst_idx, :]
    y_trn, y_tst = target[trn_idx], target[tst_idx]

    modelK = train(x_train=x_trn, y_train=y_trn, epochs=1000, lr=0.1)
    with torch.no_grad():
        y_predicted = modelK(torch.from_numpy(x_tst).float()).numpy()

    rmse, mape, mae, mbe, r2 = error_metrics(y_tst, y_predicted)
    rmse_all[k] = rmse
    mape_all[k] = mape
    mae_all[k] = mae
    mbe_all[k] = mbe
    r2_all[k] = r2

rmse_lr_cv = rmse_all.mean()
mape_lr_cv = mape_all.mean()
mae_lr_cv = mae_all.mean()
mbe_lr_cv = mbe_all.mean()
r2_lr_cv = r2_all.mean()

rmse_all = np.zeros((K, 1))
mape_all = np.zeros((K, 1))
mae_all = np.zeros((K, 1))
mbe_all = np.zeros((K, 1))
r2_all = np.zeros((K, 1))
for k in range(K):
    tst_idx = splits[k]
    trn_idx = np.delete(idx, splits[k])
    x_trn, x_tst = data[trn_idx, :], data[tst_idx, :]
    y_trn, y_tst = target[trn_idx], target[tst_idx]

    modelK = train(x_train=x_trn, y_train=y_trn, epochs=1000, lr=0.1, ridge=True, reg_lambda=0.5)
    with torch.no_grad():
        y_predicted = modelK(torch.from_numpy(x_tst).float()).numpy()

    rmse, mape, mae, mbe, r2 = error_metrics(y_tst, y_predicted)
    rmse_all[k] = rmse
    mape_all[k] = mape
    mae_all[k] = mae
    mbe_all[k] = mbe
    r2_all[k] = r2

rmse_rdg_cv = rmse_all.mean()
mape_rdg_cv = mape_all.mean()
mae_rdg_cv = mae_all.mean()
mbe_rdg_cv = mbe_all.mean()
r2_rdg_cv = r2_all.mean()

rmse_all = np.zeros((K, 1))
mape_all = np.zeros((K, 1))
mae_all = np.zeros((K, 1))
mbe_all = np.zeros((K, 1))
r2_all = np.zeros((K, 1))
for k in range(K):
    tst_idx = splits[k]
    trn_idx = np.delete(idx, splits[k])
    x_trn, x_tst = data[trn_idx, :], data[tst_idx, :]
    y_trn, y_tst = target[trn_idx], target[tst_idx]

    modelK = train(x_train=x_trn, y_train=y_trn, epochs=1000, lr=0.1, lasso=True, reg_lambda=5)
    with torch.no_grad():
        y_predicted = modelK(torch.from_numpy(x_tst).float()).numpy()

    rmse, mape, mae, mbe, r2 = error_metrics(y_tst, y_predicted)
    rmse_all[k] = rmse
    mape_all[k] = mape
    mae_all[k] = mae
    mbe_all[k] = mbe
    r2_all[k] = r2

rmse_lso_cv = rmse_all.mean()
mape_lso_cv = mape_all.mean()
mae_lso_cv = mae_all.mean()
mbe_lso_cv = mbe_all.mean()
r2_lso_cv = r2_all.mean()

print('Cross-Validated RMSE')
print('Lin Reg: {}'.format(rmse_lr_cv))
print('Rdg Reg: {}'.format(rmse_rdg_cv))
print('Lasso: {}'.format(rmse_lso_cv))

print('Cross-Validated MAPE')
print('Lin Reg: {}'.format(mape_lr_cv))
print('Rdg Reg: {}'.format(mape_rdg_cv))
print('Lasso: {}'.format(mape_lso_cv))

print('Cross-Validated MAE')
print('Lin Reg: {}'.format(mae_lr_cv))
print('Rdg Reg: {}'.format(mae_rdg_cv))
print('Lasso: {}'.format(mae_lso_cv))

print('Cross-Validated MBE')
print('Lin Reg: {}'.format(mbe_lr_cv))
print('Rdg Reg: {}'.format(mbe_rdg_cv))
print('Lasso: {}'.format(mbe_lso_cv))

print('Cross-Validated R^2')
print('Lin Reg: {}'.format(r2_lr_cv))
print('Rdg Reg: {}'.format(r2_rdg_cv))
print('Lasso: {}'.format(r2_lso_cv))
