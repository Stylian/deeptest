import random as rd
import numpy as np
import math

# 1 1 1 1
# 0 1 0 0
# 0 0 1 0
# 1 0 1 1
# 0 1 1 ?


def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


sigmoid = np.vectorize(lambda t: sigmoid_function(t))
sigmoid_deriv = np.vectorize(lambda t: sigmoid_function(t) - (1-sigmoid_function(t)))


def feed_forward(w, x):
    sum1 = np.dot(x, w)
    return sigmoid(sum1)


def back_propagation(y_out, y, w, x):
    errors = y_out - y
    delta = errors * sigmoid_deriv(y)
    return w + np.dot(np.transpose(x), delta)


x = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1]
]

y_out = [1, 0, 0, 1]

w = [rd.random(), rd.random(), rd.random()]

for i in range(10000):
    y = feed_forward(w, x)
    w = back_propagation(y_out, y, w, x)

print(w)

final_problem = feed_forward(w, [0, 1, 1])
print(final_problem)