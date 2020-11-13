# Homework 3 Solutions
# Hannah Fontenot

# NOTE: This is not the only way to solve the HW3 problems; this is an example.

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from math import floor, ceil, log10

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

"""
Part 1: Regression
Diabetes dataset
"""

diabetes = datasets.load_diabetes()
print(diabetes.DESCR)

data = diabetes.data
target = diabetes.target
names = diabetes.feature_names

target = target.reshape(-1, 1)

"""
Problem 1: Explore some of the relationships between features
"""

R = []
print('Correlation between features and target:')
for i, feature in enumerate(names):
    r = np.corrcoef(data[:, i], target[:, 0])[0, 1]
    if abs(r) > 0.4:
        R.append(i)
    s = '*' if abs(r) > 0.4 else ''
    print(feature.ljust(5, ' '), '\t', r, '\t', s)

# Plot the features with correlation > 0.4
print('There are {} features with correlation >0.4 with target'.format(len(R)))
plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(data[:, R[i]], target, 'o')
    plt.xlabel(names[R[i]], fontsize=12)
    plt.ylabel('Target', fontsize=12)
    plt.title('r = {}'.format(round(np.corrcoef(data[:, R[i]], target[:, 0])[0, 1], 3), fontsize=14))
plt.show()

plt.figure()
plt.hist(target, color='g', bins=20)
plt.xlabel('Target value')
plt.ylabel('Frequency')
plt.title('Target value distribution')
plt.show()


"""
Problem 2: Train regression model w/ cross-validation
"""

# Define a deep neural network
class ThreeLayerNetwork(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        relu1 = self.linear1(x).clamp(min=0)    # first layer: ReLU activation
        relu2 = self.linear2(relu1).clamp(min=0)    # second layer: ReLU activation
        out = self.linear3(relu2)   # output layer: linear activation
        return out


# Training function
def train(x_train, y_train, task, H1, H2, epochs=500, lr=1e-3, nclass=None, verbose=True):

    # Select regression or classification for model instantiation and loss function
    if 'REG' in task.upper():
        model = ThreeLayerNetwork(x_train.shape[1], H1, H2, y_train.shape[1])
        criterion = torch.nn.MSELoss()
    if 'CLASS' in task.upper():
        model = ThreeLayerNetwork(x_train.shape[1], H1, H2, nclass)
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        inputs = torch.from_numpy(x_train).float()
        labels = torch.from_numpy(y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        # Loss is calculated differently for regression vs classification
        if 'REG' in task.upper():
            loss = criterion(outputs, labels)
        if 'CLASS' in task.upper():
            loss = criterion(outputs, labels.squeeze().long())
        loss.backward()

        optimizer.step()

        if verbose:
            if epoch % 25 == 0:
                print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model


# Cross-validated training:
task = 'regression'
H1 = H2 = 10

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=False)

rmse_cv, r2_cv = np.empty(0), np.empty(0)

for trn_idx, tst_idx in kf.split(data):
    x_trn, y_trn = data[trn_idx, :], target[trn_idx, :]
    x_tst, y_tst = data[tst_idx, :], target[tst_idx, :]

    # Train model
    model = train(x_trn, y_trn, task, H1, H2, 2000)

    # Test model
    with torch.no_grad():
        y_pred = model(torch.from_numpy(x_tst).float()).numpy()

    # Calculate rmse, r2
    rmse_cv = np.append(rmse_cv, np.sqrt(np.mean((y_pred - y_tst) ** 2)))
    r2_cv = np.append(r2_cv, np.corrcoef(y_pred.squeeze(), y_tst.squeeze())[0, 1] ** 2)

# Cross-validated RMSE, R2
print('Cross-validated RMSE: {}, R2: {}'.format(rmse_cv.mean(), r2_cv.mean()))

# Train/test on full dataset
print(data.shape)
target = target.reshape(-1, 1)
x_train, y_train = data[:400, ], target[:400, ]
x_test, y_test = data[400:, ], target[400:, ]

model = train(x_train, y_train, task, H1, H2, 2000)
with torch.no_grad():
    y_pred = model(torch.from_numpy(x_test).float()).numpy()
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = np.corrcoef(y_pred.squeeze(), y_test.squeeze())[0, 1] ** 2
print('RMSE: {}, R2: {}'.format(rmse, r2))

# Plots
plt.figure()
plt.plot(y_test, y_pred, 'o', label='actual prediction')
plt.plot(list(range(floor(min(y_test)), ceil(max(y_test)))), list(range(floor(min(y_test)), ceil(max(y_test)))), label='perfect prediction')
plt.legend()
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('DNN for Regression')

"""
Problem 3: Regression model hyperparameter tuning
"""

H1_list = list(range(2, 11))  # NOTE: This will take a long time because of the number of hyperparameter combos to run!!
H2_list = list(range(2, 11))
lr_list = [1e-4, 1e-3, 1e-2]

rmse_trn = np.zeros((len(H1_list), len(H2_list), len(lr_list)))
rmse_tst = np.zeros_like(rmse_trn)

for h1, H1 in enumerate(H1_list):
    for h2, H2 in enumerate(H2_list):
        for lr, LR in enumerate(lr_list):
            rmse_trn_cv, rmse_tst_cv = np.empty(0), np.empty(0)
            for trn_idx, tst_idx in kf.split(data):
                x_trn, y_trn = data[trn_idx, :], target[trn_idx, :]
                x_tst, y_tst = data[tst_idx, :], target[tst_idx, :]

                # Train model
                model = train(x_trn, y_trn, task, H1, H2, 5000, LR, verbose=False)

                # Test model
                with torch.no_grad():
                    y_pred_trn = model(torch.from_numpy(x_trn).float()).numpy()
                    y_pred_tst = model(torch.from_numpy(x_tst).float()).numpy()

                # Calculate training/testing RMSE
                rmse_trn_cv = np.append(rmse_trn_cv, np.sqrt(np.mean((y_pred_trn - y_trn) ** 2)))
                rmse_tst_cv = np.append(rmse_tst_cv, np.sqrt(np.mean((y_pred_tst - y_tst) ** 2)))

            rmse_trn[h1, h2, lr] = rmse_trn_cv.mean()
            rmse_tst[h1, h2, lr] = rmse_tst_cv.mean()

            print('H1 = {}, H2 = {}, LR = {}: Training RMSE = {}, Testing RMSE = {}'.format(H1, H2, LR,  rmse_trn_cv.mean(), rmse_tst_cv.mean()))

# Find best hyperparameter values
i, j, k = np.argwhere(rmse_tst == np.min(rmse_tst))[0]
h1_best, h2_best, lr_best = H1_list[i], H2_list[j], lr_list[k]

# Train/test again
model = train(x_train, y_train, task, h1_best, h2_best, lr=lr_best, epochs=2000)
with torch.no_grad():
    y_pred = model(torch.from_numpy(x_test).float()).numpy()
rmse_tuned = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_tuned = np.corrcoef(y_pred.squeeze(), y_test.squeeze())[0, 1] ** 2
print('Tuned RMSE: {}, Tuned R2: {}'.format(rmse_tuned, r2_tuned))

# Overfitting plots
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(H1_list, rmse_trn[:, 0, 0], label='Training RMSE')
plt.plot(H1_list, rmse_tst[:, 0, 0], label='Validation RMSE')
plt.legend()
plt.xlabel('H1')
plt.ylabel('RMSE')
plt.title('H1 neurons')
plt.subplot(1, 3, 2)
plt.plot(H2_list, rmse_trn[0, :, 0], label='Training RMSE')
plt.plot(H2_list, rmse_tst[0, :, 0], label='Validation RMSE')
plt.legend()
plt.xlabel('H2')
plt.ylabel('RMSE')
plt.title('H2 neurons')
plt.subplot(1, 3, 3)
plt.plot([log10(x) for x in lr_list], rmse_trn[0, 0, :], label='Training RMSE')
plt.plot([log10(x) for x in lr_list], rmse_tst[0, 0, :], label='Validation RMSE')
plt.legend()
plt.xlabel('Learning Rate (log)')
plt.ylabel('RMSE')
plt.title('Learning rate')


"""
Part 2: Classification
Wine dataset
"""

wine = datasets.load_wine()
print(wine.DESCR)

data = wine.data
target = wine.target
names = wine.feature_names
nclass = len(np.unique(target))

target = target.reshape(-1, 1)

# Shuffle data
idx = list(range(len(target)))
np.random.shuffle(idx)
data = data[idx, :]
target = target[idx, :]


"""
Problem 4: Explore some of the relationships between features
"""

R = []
print('Correlation between features and target:')
for i, feature in enumerate(names):
    r = np.corrcoef(data[:, i], target[:, 0])[0, 1]
    if abs(r) > 0.5:
        R.append(i)
    s = '*' if abs(r) > 0.5 else ''
    print(feature.ljust(27, ' '), '\t', r, '\t', s)

# Plot the features with correlation > 0.5
print('There are {} features with correlation >0.5 with target'.format(len(R)))
plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.plot(data[:, R[i]], target, 'o')
    plt.xlabel(names[R[i]], fontsize=12)
    plt.ylabel('Target', fontsize=12)
    plt.title('r = {}'.format(round(np.corrcoef(data[:, R[i]], target[:, 0])[0, 1], 3), fontsize=14))
plt.show()

plt.figure()
plt.hist(target, color='g', bins=20)
plt.xlabel('Target value')
plt.ylabel('Frequency')
plt.title('Target value distribution')
plt.show()

plt.figure()
plt.plot(data[:, 6][target[:, 0] == 0], data[:, 11][target[:, 0] == 0], 'o')
plt.plot(data[:, 6][target[:, 0] == 1], data[:, 11][target[:, 0] == 1], 'x')
plt.plot(data[:, 6][target[:, 0] == 2], data[:, 11][target[:, 0] == 2], '^')
plt.xlabel(names[6])
plt.ylabel(names[11])
plt.legend(['Class 0', 'Class 1', 'Class 2'])
plt.show()


"""
Problem 5: Train classification model w/ cross-validation
"""

# Cross-validated training:
task = 'classification'
H1 = H2 = 10

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

acc_cv, cf_cv = np.empty(0), []

for trn_idx, tst_idx in kf.split(data):
    x_trn, y_trn = data[trn_idx, :], target[trn_idx, :]
    x_tst, y_tst = data[tst_idx, :], target[tst_idx, :]

    # Train model
    model = train(x_trn, y_trn, task, H1, H2, nclass=nclass, epochs=2000)

    # Test model
    with torch.no_grad():
        y_pred = model(torch.from_numpy(x_tst).float())
    _, y_pred = torch.max(y_pred, 1)
    y_pred = y_pred.numpy().reshape(-1, 1)

    # Calculate rmse, r2
    acc_cv = np.append(acc_cv, sum(y_pred == y_tst) / len(y_pred))
    cf_cv.append(confusion_matrix(y_tst, y_pred))

# Cross-validated accuracy, confusion matrix
print('Cross-validated Accuracy: {}, \nCross-validated Confusion Matrix: \n{}'.format(acc_cv.mean(), sum(cf_cv)/len(cf_cv)))
plt.figure()
ax = sns.heatmap(sum(cf_cv)/len(cf_cv), cbar=False, annot=True, annot_kws={'fontsize': 20})
ax.set_ylim([0, 3])
ax.invert_yaxis()
plt.xlabel('Y_test', fontsize=20)
plt.ylabel('Y_pred', fontsize=20)

# Train/test on full dataset
print(data.shape)
target = target.reshape(-1, 1)
x_train, y_train = data[:150, ], target[:150, ]
x_test, y_test = data[150:, ], target[150:, ]

model = train(x_train, y_train, task, H1, H2, nclass=nclass, epochs=2000)
with torch.no_grad():
    y_pred = model(torch.from_numpy(x_test).float())
_, y_pred = torch.max(y_pred, 1)
y_pred = y_pred.numpy().reshape(-1, 1)
acc = sum(1*(y_pred == y_test)) / len(y_pred)
cf = confusion_matrix(y_test, y_pred)
print('Accuracy: {}, \nConfusion Matrix: \n'.format(acc), cf)

# Plots
plt.figure()
plt.plot(np.sort(y_pred, axis=None), 'bo', markersize=10, label='Predicted')
plt.plot(np.sort(y_test, axis=None), 'rx', markersize=10, label='Measured')
plt.legend()

"""
Problem 6: Classification model hyperparameter tuning
"""

H1_list = list(range(2, 11))  # NOTE: This will take a long time because of the number of hyperparameter combos to run!!
H2_list = list(range(2, 11))
lr_list = [1e-4, 1e-3, 1e-2]

acc_trn = np.zeros((len(H1_list), len(H2_list), len(lr_list)))
acc_tst = np.zeros_like(acc_trn)

for h1, H1 in enumerate(H1_list):
    for h2, H2 in enumerate(H2_list):
        for lr, LR in enumerate(lr_list):
            acc_trn_cv, acc_tst_cv = np.empty(0), np.empty(0)
            for trn_idx, tst_idx in kf.split(data):
                x_trn, y_trn = data[trn_idx, :], target[trn_idx, :]
                x_tst, y_tst = data[tst_idx, :], target[tst_idx, :]

                # Train model
                model = train(x_trn, y_trn, task, H1, H2, nclass=nclass, epochs=2000, verbose=False)

                # Test model
                with torch.no_grad():
                    y_pred_trn = model(torch.from_numpy(x_trn).float())
                    y_pred_tst = model(torch.from_numpy(x_tst).float())
                _, y_pred_trn = torch.max(y_pred_trn, 1)
                _, y_pred_tst = torch.max(y_pred_tst, 1)
                y_pred_trn = y_pred_trn.numpy().reshape(-1, 1)
                y_pred_tst = y_pred_tst.numpy().reshape(-1, 1)
                acc = sum(1 * (y_pred == y_test)) / len(y_pred)

                # Calculate training/testing RMSE
                acc_trn_cv = np.append(acc_trn_cv, sum(1 * (y_pred_trn == y_trn)) / len(y_pred_trn))
                acc_tst_cv = np.append(acc_tst_cv, sum(1 * (y_pred_tst == y_tst)) / len(y_pred_tst))

            acc_trn[h1, h2, lr] = acc_trn_cv.mean()
            acc_tst[h1, h2, lr] = acc_tst_cv.mean()

            print('H1 = {}, H2 = {}, LR = {}: Training Accuracy = {}, Testing Accuracy = {}'.format(H1, H2, LR,  acc_trn_cv.mean(), acc_tst_cv.mean()))

# Find best hyperparameter values
i, j, k = np.argwhere(acc_tst == np.max(acc_tst))[0]  # NOTE: Find the highest accuracy (as opposed to lowest RMSE for regression)
h1_best, h2_best, lr_best = H1_list[i], H2_list[j], lr_list[k]

# Train/test again
model = train(x_train, y_train, task, h1_best, h2_best, nclass=nclass, lr=lr_best, epochs=2000)
with torch.no_grad():
    y_pred = model(torch.from_numpy(x_test).float())
_, y_pred = torch.max(y_pred, 1)
y_pred = y_pred.numpy().reshape(-1, 1)
acc_tuned = sum(1 * (y_pred == y_test)) / len(y_pred)
cf_tuned = confusion_matrix(y_test, y_pred)
print('Tuned Accuracy: {}, \nTuned Confusion Matrix: \n'.format(acc_tuned), cf_tuned)

# Overfitting plots
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(H1_list, acc_trn[:, 0, 0], label='Training Accuracy')
plt.plot(H1_list, acc_tst[:, 0, 0], label='Validation Accuracy')
plt.legend()
plt.xlabel('H1')
plt.ylabel('Accuracy')
plt.title('H1 neurons')
plt.subplot(1, 3, 2)
plt.plot(H2_list, acc_trn[0, :, 0], label='Training Accuracy')
plt.plot(H2_list, acc_tst[0, :, 0], label='Validation Accuracy')
plt.legend()
plt.xlabel('H2')
plt.ylabel('Accuracy')
plt.title('H2 neurons')
plt.subplot(1, 3, 3)
plt.plot([log10(x) for x in lr_list], acc_trn[0, 0, :], label='Training Accuracy')
plt.plot([log10(x) for x in lr_list], acc_tst[0, 0, :], label='Validation Accuracy')
plt.legend()
plt.xlabel('Learning Rate (log)')
plt.ylabel('Accuracy')
plt.title('Learning rate')
