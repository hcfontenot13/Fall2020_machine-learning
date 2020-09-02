# Use Alt + Shift + E to run a single line or highlighted selection
print('Hello World')

import numpy as np

# Array creation
a = np.array([0, 1, 2, 3, 4])
print(a)

b = np.array([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
print(b)

# Attributes
print(a.shape)
print(a.ndim)
print(a.size)
print(b.shape)
print(b.ndim)
print(b.size)

# Methods
print(a)
print(a.sum())

print(b)
print(b.sum())          # No argument: Returns sum of all values
print(b.sum(axis=0))    # Axis 0: Returns sum by columns
print(b.sum(axis=1))    # Axis 1: Returns sum by rows

print(a.mean())
print(b.mean(axis=0))
print(b.mean(axis=1))

print(a.min())
print(a.argmin())

print(a.cumsum())

# Built-in functions
c = np.ones(3)
print(c)
print(4.5 * c)

d = np.zeros((3, 3))
print(d)

e = np.eye(3)
print(e)

f = np.concatenate([a, a], axis=0)
print(f)

g = np.concatenate([a, a], axis=1)  # Error: axis out of bounds
print(a.ndim)

a1 = np.expand_dims(a, axis=1)
print(a1)
print(a1.shape)

g = np.concatenate([a1, a1], axis=1)
print(g)

print(a)
print(np.count_nonzero(a > 2))
print(b)
print(np.count_nonzero(b >= 9))

# Random
print(np.random.random())           # No argument: Output is a single value
print(np.random.random(5))          # Integer argument: Output is a vector
print(np.random.random((5, 5)))     # Tuple argument: Output is an array

print(np.random.randint())          # Error: needs at least one argument
print(np.random.randint(10))        # One argument: Upper bound
print(np.random.randint(20, 50))    # Two arguments: Lower, upper bounds
print(np.random.randint(0, 100, (1, 5)))  # Three arguments: Lower, upper bounds, size of output

print(np.random.randn())        # No argument: Output is a single value
print(np.random.randn(5))       # Integer argument: Output is a vector
print(np.random.randn(5, 5))    # Multiple arguments: Output is an array

np.random.seed(1)   # Setting a seed generates the same "random" numbers each time
np.random.randn(1000).mean()

v1 = np.random.randn(1000)  # Standard normal
v1.mean()
v1.std()

from matplotlib import pyplot as plt

plt.figure()
plt.hist(v1, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Standard normal: $\mu=0, \sigma^2=1$')
plt.show()

mu = 5
sigma = 2
v2 = sigma * v1 + mu
v2.mean()
v2.std()

plt.figure()
plt.hist(v2, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('$\mu=5, \sigma^2=2$')
plt.show()

v3 = np.random.choice(v2, 100, replace=True)
v3.mean()
v3.std()

plt.figure()
plt.hist(v3, bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('A small sample with replacement')
plt.show()

# Exercise: Rolling dice
'''Roll a die 100 times and count the number of times it lands on each number'''
np.random.seed(5)
rolls = np.random.randint(1, 7, (100, 1))   # Note upper bound is 7 instead of 6
n1 = np.count_nonzero(rolls == 1)
n = [np.count_nonzero(rolls == x) for x in range(1, 7)]
p = n / 100     # TypeError
p = np.array(n) / 100
