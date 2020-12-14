# Homework 4 Solutions
# Hannah Fontenot

# NOTE: This is not the only way to solve the HW4 problems; this is an example.

import pandas as pd
import numpy as np
import os

import torch
from sklearn import svm

from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


""" Read data """

filename = os.path.join(os.getcwd(), 'Homework4Data.csv')
df = pd.read_csv(filename, sep=',')


""" Preprocessing """

# Rename date/time column
df.rename(columns={'Date/Time': 'Date_Time'}, inplace=True)

# Check for NA & fill in
print(df.isnull().sum())
idx = np.where(df['AirTemp'].isnull())[0].tolist()  # Find index where AirTemp is NA
for i in idx:                                         # Replace with mean of column
    df['AirTemp'][i] = df['AirTemp'].mean()
idx = np.where(df['WindDirection'].isnull())[0].tolist()  # Find index where WindDirection is NA
for i in idx:                                               # Replace with mean of column
    df['WindDirection'][i] = df['WindDirection'].mean()
idx = np.where(df['SeaLvlPressure'].isnull())[0].tolist()  # Find index where SeaLvlPressure is NA
for i in idx:                                                # Replace with mean of column
    df['SeaLvlPressure'][i] = df['SeaLvlPressure'].mean()

# Check for uniqueness
print(df.nunique())

# Pick a season with stable temperature
plt.plot(df['AirTemp'])
plt.plot(df['ElectricityUse'])
data = df[:700]

# Split into Weekdays/Weekends
date_time = [datetime.strptime(x, "%m/%d/%Y %H:%M") for x in data['Date_Time']]
data['Date_Time'] = pd.Series(date_time)
data['Weekday'] = pd.Series([x.isoweekday() for x in data['Date_Time']])

# Check which weekdays to keep
plt.subplot(2, 1, 1)
plt.plot(data['ElectricityUse'])
plt.subplot(2, 1, 2)
plt.plot(data['Weekday'])

# Keep weekdays 2-6
data_wkdy = data[(data['Weekday'] >= 2) & (data['Weekday'] <= 6)]
data_wknd = data[(data['Weekday'] < 2) & (data['Weekday'] > 6)]

# Feature selection
features = ['AirTemp', 'SeaLvlPressure', 'WindDirection', 'WindSpeed']
print('Correlation between features and target:')
for feature in features:
    r = np.corrcoef(data_wkdy[feature], data_wkdy['ElectricityUse'])[0, 1]
    s = '*' if abs(r) > 0.2 else ''
    print(feature.ljust(5, ' '), '\t', r, '\t', s)

# Plot feature + target
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(data_wkdy['AirTemp'].tolist(), 'g')
plt.title('Air Temperature')
plt.subplot(2, 1, 2)
plt.plot(data_wkdy['ElectricityUse'].tolist(), 'b')
plt.title('Electricity Use')


""" Data Preparation """

# Split into train/test
one_day = 24
test_size = 2 * one_day
data_train = data_wkdy[:-test_size][['AirTemp', 'ElectricityUse']].to_numpy()
data_test = data_wkdy[-test_size:][['AirTemp', 'ElectricityUse']].to_numpy()


def make_sequences(input_data, sequence_len):
    inout_seq = []
    L = len(input_data)
    for i in range(L - sequence_len):
        train_seq = input_data[i:i + sequence_len]
        train_label = input_data[i + sequence_len:i + sequence_len + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


""" RNN/LSTM model definition """


class RNN(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.rnn = torch.nn.RNN(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = torch.zeros(nLayers, 1, D_hidden)

    def forward(self, input_seq):
        rnn_out, self.hidden = self.rnn(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(rnn_out.view(len(input_seq), -1))
        return y_pred[-1]


class LSTM(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.lstm = torch.nn.LSTM(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = (torch.zeros(nLayers, 1, D_hidden),
                       torch.zeros(nLayers, 1, D_hidden))

    def forward(self, input_seq):
        lstm_out, self.hidden = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(lstm_out.view(len(input_seq), -1))
        return y_pred[-1]


""" Training and testing """


def train(training_sequences, rnn_type, num_hid=10, num_layers=1, learning_rate=1e-3, epochs=100):
    D_in = training_sequences[0][0].shape[1]
    D_out = training_sequences[0][1].shape[1]
    D_hidden = num_hid

    if rnn_type.upper() == 'RNN':
        model = RNN(D_in, D_hidden, D_out, nLayers=num_layers)
    elif rnn_type.upper() == 'LSTM':
        model = LSTM(D_in, D_hidden, D_out, nLayers=num_layers)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        ep_loss = 0
        for seq, labels in training_sequences:

            optimizer.zero_grad()

            if rnn_type.upper() == 'LSTM':
                model.hidden = (torch.zeros(model.nLayers, 1, model.D_hidden),
                                torch.zeros(model.nLayers, 1, model.D_hidden))
            else:
                model.hidden = torch.zeros(model.nLayers, 1, model.D_hidden)

            outputs = model(seq)
            loss = criterion(outputs[1], labels[0, 1])
            loss.backward()

            optimizer.step()

            ep_loss += loss.item()

        if epoch % 10 == 1:
            print('epoch {}: loss = {}'.format(epoch, ep_loss))

    return model


def test(model, rnn_type, test_inputs, seq_len, pred_len):
    model.eval()
    for it in range(pred_len):
        test_seq = torch.FloatTensor(test_inputs[it:it + seq_len])
        with torch.no_grad():
            if rnn_type.upper() == 'LSTM':
                model.hidden = (torch.zeros(model.nLayers, 1, model.D_hidden),
                                torch.zeros(model.nLayers, 1, model.D_hidden))
            else:
                model.hidden = torch.zeros(model.nLayers, 1, model.D_hidden)
            test_inputs.append(model(test_seq).tolist())
    return test_inputs


""" Cross-validation """


def cross_val(training_data, k, rnn_type, hyperparameter_dict, epochs):
    k_folds = k
    k_splits = np.array_split(training_data, k_folds)
    k_rmse = []
    k_mape = []

    num_hid = hyperparameter_dict['num_hidden']
    num_layer = hyperparameter_dict['num_layers']
    lr = hyperparameter_dict['learning_rate']
    seq_len = hyperparameter_dict['sequence_len']

    scaler = MinMaxScaler(feature_range=(-1, 1))

    for k, k_data in enumerate(k_splits):
        print('FOLD {} OUT OF {}'.format(k+1, k_folds))

        # Split into train/test sets and normalize
        idx_trn = range(0, k_data.shape[0] - one_day)
        idx_tst = range(k_data.shape[0] - one_day, k_data.shape[0])

        k_trn = k_data[idx_trn]
        k_tst = k_data[idx_tst]

        k_trn_norm = torch.FloatTensor(scaler.fit_transform(k_trn))
        k_tst_norm = torch.FloatTensor(scaler.fit_transform(k_tst))

        # seq_len = one_day  # 24 hours
        trn_sequences = make_sequences(k_trn_norm, seq_len)
        tst_sequences = k_tst_norm.tolist()

        # Train/test
        mdl = train(training_sequences=trn_sequences, rnn_type=rnn_type, num_hid=num_hid, num_layers=num_layer, learning_rate=lr, epochs=epochs)
        y_pred = np.array(test(mdl, rnn_type, tst_sequences, seq_len, one_day))

        # Unscale
        y_pred = scaler.inverse_transform(np.array(y_pred))[one_day:one_day+k_tst.shape[0], -1]

        # RMSE, MAPE
        rmse = np.sqrt(np.mean((y_pred - k_tst[-one_day:, -1]) ** 2))
        mape = np.mean(np.abs((y_pred - k_tst[-one_day:, -1]) / k_tst[-one_day:, -1]))
        k_rmse.append(rmse)
        k_mape.append(mape)

    print('Cross-validated RMSE: {}'.format(np.array(k_rmse).mean()))
    print('Cross-validated MAPE: {}'.format(np.array(k_mape).mean()))

    return np.array(k_rmse).mean(), np.array(k_mape).mean()


""" Hyperparameter tuning """

# Hyperparameters to tune:
# Number of hidden layer neurons
# Number of hidden layers
# Learning rate
# Sequence length


def tune_hyperparameters(training_data, k_folds, rnn_type, num_hidden_list, num_layers_list, learning_rate_list, sequence_len_list, epochs):
    total_combos = len(num_hidden_list) * len(num_layers_list) * len(learning_rate_list) * len(sequence_len_list)

    # Keep track of results
    run = 0
    cross_val_results = {}

    # Run a grid search, recording the results
    for v1 in range(len(num_hidden_list)):
        for v2 in range(len(num_layers_list)):
            for v3 in range(len(learning_rate_list)):
                for v4 in range(len(sequence_len_list)):
                    hyp_dict = {
                        'num_hidden': num_hidden_list[v1],
                        'num_layers': num_layers_list[v2],
                        'learning_rate': learning_rate_list[v3],
                        'sequence_len': sequence_len_list[v4]
                    }
                    print('COMBINATION {} OUT OF {}'.format(run, total_combos))
                    rmse, mape = cross_val(training_data=training_data, k=k_folds, rnn_type=rnn_type, hyperparameter_dict=hyp_dict, epochs=epochs)
                    cross_val_results[run] = {
                        'parameters': hyp_dict,
                        'rmse': rmse,
                        'mape': mape
                    }
                    run += 1

    # Find best results
    all_rmse = {i: cross_val_results[i]['rmse'] for i in range(total_combos)}
    all_mape = {i: cross_val_results[i]['mape'] for i in range(total_combos)}

    best_rmse = [(k, v) for k, v in all_rmse.items() if v == min(all_rmse.values())]
    best_mape = [(k, v) for k, v in all_mape.items() if v == min(all_mape.values())]

    # Choose one error metric to be used to determine best hyperparameters
    best_run = best_mape[0][0]
    best_hyperparameters = cross_val_results[best_run]['parameters']

    return best_mape, best_hyperparameters


""" Plot Results """


def plot_results(predicted, measured, rnn_type, mape):
    plt.figure()
    plt.plot(measured, linewidth=2, label='Measured')
    plt.plot(predicted, '-o', linewidth=2, label='Predicted')
    plt.legend(fontsize=14)
    plt.title('{} Network: MAPE = {}%'.format(rnn_type.upper(), np.round(mape * 100, 3)), fontsize=18)
    plt.xlabel('Time of day', fontsize=16)
    plt.ylabel('Energy Use (W)', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(ticks=range(0, len(predicted), int(one_day/4)),
               labels=int(len(predicted)/one_day)*list(range(0, 24, int(one_day/4))),
               fontsize=14)
    plt.show()


""" Total Workflow """

# Values of hyperparameters
num_hidden_list = [10, 20]
num_layers_list = [1, 2]
learning_rate_list = [1e-3, 1e-2]
sequence_len_list = [24, 12]

scaler = MinMaxScaler(feature_range=(-1, 1))
data_train_norm = torch.FloatTensor(scaler.fit_transform(data_train))
data_test_norm = torch.FloatTensor(scaler.fit_transform(data_test))

mdl_type = 'LSTM'  # Run this for both RNN and LSTM

best_mape, best_hyperparameters = tune_hyperparameters(
    training_data=data_train, k_folds=2, rnn_type=mdl_type,
    num_hidden_list=num_hidden_list, num_layers_list=num_layers_list,
    learning_rate_list=learning_rate_list, sequence_len_list=sequence_len_list,
    epochs=5)

num_hid = best_hyperparameters['num_hidden']
num_layer = best_hyperparameters['num_layers']
lr = best_hyperparameters['learning_rate']
seq_len = best_hyperparameters['sequence_len']

trn_sequences = make_sequences(data_train_norm, seq_len)
tst_sequences = data_test_norm.tolist()

# Train/test
mdl = train(training_sequences=trn_sequences, rnn_type=mdl_type, num_hid=num_hid, num_layers=num_layer, learning_rate=lr)
y_pred = np.array(test(mdl, mdl_type, tst_sequences, seq_len, one_day))

# Unscale
y_pred = scaler.inverse_transform(np.array(y_pred))[one_day:one_day + data_test.shape[0], -1]

# RMSE, MAPE
rmse = np.sqrt(np.mean((y_pred - data_test[:, -1]) ** 2))
mape = np.mean(np.abs((y_pred - data_test[:, -1]) / data_test[:, -1]))

# Plot
plot_results(y_pred, data_test[:, -1], mdl_type, mape)


""" SVM """

data_train_svr = data_train
data_train_svr[1:, 1] = data_train_svr[:-1, 1]

mdl = svm.SVR(kernel='rbf')
mdl.fit(data_train[:, :-1], data_train[:, -1])
y_pred1 = mdl.predict(data_train[:, :-1])
y_pred2 = mdl.predict(data_test[:, :-1])

mape1 = np.mean(np.abs((y_pred1 - data_train[:, -1]) / data_train[:, -1]))
mape2 = np.mean(np.abs((y_pred2 - data_test[:, -1]) / data_test[:, -1]))
print('SVM untuned training MAPE: {}'.format(mape1))
print('SVM untuned testing MAPE: {}'.format(mape2))

plot_results(y_pred2, data_test[:, -1], 'SVM', mape2)

param_grid = {
    'kernel': ('linear', 'rbf'),
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}
grid = GridSearchCV(svm.SVR(), param_grid, refit=True, verbose=3)
grid.fit(data_train[:, 0].reshape(-1, 1), data_train[:, -1])

print('Best parameters: {}'.format(grid.best_params_))
print('Best estimator: {}'.format(grid.best_estimator_))

y_pred1 = grid.predict(data_train[:, :-1])
y_pred2 = grid.predict(data_test[:, :-1])
mape1 = np.mean(np.abs((y_pred1 - data_train[:, -1]) / data_train[:, -1]))
mape2 = np.mean(np.abs((y_pred2 - data_test[:, -1]) / data_test[:, -1]))
print('SVM tuned training MAPE: {}'.format(mape1))
print('SVM tuned testing MAPE: {}'.format(mape2))

plot_results(y_pred2, data_test[:, -1], 'SVM', mape2)
