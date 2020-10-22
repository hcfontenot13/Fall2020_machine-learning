import torch


device = torch.device('cpu')


# x -> w1 -> REUL -> w2 -> y
N, D_in, H, D_out = 2, 5, 6, 8
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)
w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)
learning_rate = 1e-4

# Manual backpropagation
for t in range(500):
    # Forward pass
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()

    # Backward pass
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# Autograd
N, D_in, H, D_out = 2, 5, 6, 8
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)
learning_rate = 1e-4

for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()

    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()


# NN module
N, D_in, H, D_out = 2, 5, 6, 8
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
learning_rate = 1e-4

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

for t in range(500):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            model.zero_grad()


# Use optimizer
N, D_in, H, D_out = 2, 5, 6, 8
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
learning_rate = 1e-4

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()
    optimizer.step()


# Define your own NN
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 2, 5, 6, 8
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
learning_rate = 1e-4

model = TwoLayerNet(D_in, H, D_out)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()
    optimizer.step()
    print(loss.item())


# Use Tensorflow
import numpy as np
import tensorflow as tf

N, D, H = 2, 5, 6

x = tf.convert_to_tensor(np.random.randn(N, D), np.float32)
y = tf.convert_to_tensor(np.random.randn(N, D), np.float32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    H, input_shape=(D, ), activation=tf.nn.relu
))
model.add(tf.keras.layers.Dense(D))
optimizer = tf.keras.optimizers.SGD(1e-1)

losses = []
for t in range(50):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.keras.losses.MeanSquaredError()(y_pred, y)
        gradients = tape.gradient(
            loss, model.trainable_variables
        )
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )


N, D, H = 2, 5, 6

x = tf.convert_to_tensor(np.random.randn(N, D), np.float32)
y = tf.convert_to_tensor(np.random.randn(N, D), np.float32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    H, input_shape=(D, ), activation=tf.nn.relu
))
model.add(tf.keras.layers.Dense(D))
optimizer = tf.keras.optimizers.SGD(1e-1)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)
history = model.fit(x, y, epochs=50, batch_size=10, steps_per_epoch=100)