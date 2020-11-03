# Lab 6 (RNN, GRU, LSTM + plotting tutorial)
# 11/3/2020

import numpy as np
import torch
from matplotlib import pyplot as plt


# Generate data
N = 100
x = np.arange(1,N+1)
y1 = 5*np.sin(5*x) + np.random.randn(N)
y2 = 0.05*x + np.random.randn(N)
data = y1 + y2
full_data = np.vstack((y2, data))
plt.plot(data)

# Split into test/train
test_size = 10
data_train = data[:-test_size]
data_test = data[-test_size:]

plt.figure()
plt.plot(range(len(data)-test_size), data_train, label='train')
plt.plot(range(len(data)-test_size-1, len(data)-1), data_test, label='test')
plt.legend()

# Normalize data & convert to tensor
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
data_train_norm = scaler.fit_transform(data_train.reshape(-1, 1))
plt.figure()
plt.plot(data_train_norm)

data_train_norm = torch.FloatTensor(data_train_norm) #.view(-1)


# Create sequences of (input_data, output_data) tuples
def make_sequences(input_data, sl):
    inout_seq = []
    L = len(input_data)
    for i in range(L-sl):
        train_seq = input_data[i:i+sl]
        train_label = input_data[i+sl:i+sl+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# Set a training sequence length
seq_len = 10

training_sequences = make_sequences(data_train_norm, seq_len)
len(training_sequences)
training_sequences[:5]  # What does a "training sequence" look like?


# RNN model
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


# GRU model
class GRU(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.gru = torch.nn.GRU(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = torch.zeros(nLayers, 1, D_hidden)

    def forward(self, input_seq):
        gru_out, self.hidden = self.gru(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(gru_out.view(len(input_seq), -1))
        return y_pred[-1]


# LSTM model
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


# Training function
def train(training_sequences, rnn_type, num_layers, learning_rate=1e-3, epochs=100):
    D_in = training_sequences[0][0].shape[1]
    D_out = training_sequences[0][1].shape[1]
    D_hidden = 50

    if rnn_type.upper() == 'RNN':
        model = RNN(D_in, D_hidden, D_out, nLayers=num_layers)
    elif rnn_type.upper() == 'GRU':
        model = GRU(D_in, D_hidden, D_out, nLayers=num_layers)
    elif rnn_type.upper() == 'LSTM':
        model = LSTM(D_in, D_hidden, D_out, nLayers=num_layers)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for seq, labels in training_sequences:

            optimizer.zero_grad()

            if rnn_type.upper() == 'LSTM':
                model.hidden = (torch.zeros(model.nLayers, 1, model.D_hidden),
                                torch.zeros(model.nLayers, 1, model.D_hidden))
            else:
                model.hidden = torch.zeros(model.nLayers, 1, model.D_hidden)

            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        if epoch % 10 == 1:
            print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model


# Testing function
def test(model, rnn_type, test_inputs, pred_len):
    model.eval()
    for i in range(pred_len):
        test_seq = torch.FloatTensor(test_inputs[-seq_len:])
        with torch.no_grad():
            if rnn_type.upper() == 'LSTM':
                model.hidden = (torch.zeros(model.nLayers, 1, model.D_hidden),
                                torch.zeros(model.nLayers, 1, model.D_hidden))
            else:
                model.hidden = torch.zeros(model.nLayers, 1, model.D_hidden)

            test_inputs.append(model(test_seq))

    # print(np.array(test_inputs[pred_len:]))
    return test_inputs[pred_len:]


# Train, test model
rnn_type = 'lstm'
number_layers = 1
model = train(training_sequences, rnn_type, number_layers)

pred_len = test_size
test_inputs = data_train_norm[-seq_len:].tolist()
y_pred_unscaled = test(model, rnn_type, test_inputs, pred_len)

# Inverse scaling
y_pred = scaler.inverse_transform(np.array(y_pred_unscaled).reshape(-1, 1))
print(y_pred)

# RMSE
rmse = np.sqrt(np.mean((y_pred - data_test) ** 2))


# Plotting
plt.figure()
plt.plot(data, label='actual', linewidth=2)
plt.plot(range(len(data)-test_size, len(data)), y_pred, '-o', markersize=10, label='predictions')
plt.legend(loc='best', fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Y-value', fontsize=14)
plt.title('{} Method: RMSE = {}'.format(rnn_type.upper(), round(rmse, 3)), fontsize=18)
plt.show()


