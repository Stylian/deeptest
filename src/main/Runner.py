import random as rd
import numpy as np
import math

x = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# A
#y_out = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# A xor B !not working
y_out = np.array([0, 0, 1, 1, 1, 1, 0, 0])

# B and C ! fails for some
#y_out = np.array([0, 0, 0, 1, 0, 0, 0, 1])

# A or B or C
#y_out = np.array([0, 1, 1, 1, 1, 1, 1, 1])

# A and B and C
#y_out = np.array([0, 0, 0, 0, 0, 0, 0, 1])

# y_out = np.array([
#     [1, 0, 0, 1],
#     [0, 1, 0, 1],
#     [1, 0, 0, 0]
# ])

def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


sigmoid = np.vectorize(lambda t: sigmoid_function(t))
sigmoid_deriv = np.vectorize(lambda t: sigmoid_function(t) - (1 - sigmoid_function(t)))


def feed_forward(w, x):
    sum1 = np.dot(x, w)
    return sigmoid(sum1)


def back_propagation(y_out, y, w, x):
    errors = y_out - y
    delta = errors * sigmoid_deriv(y)
    return w + np.dot(np.transpose(x), delta)


w = np.array([rd.random(), rd.random(), rd.random()])
# w = np.array([
#     [rd.random(), rd.random(), rd.random()],
#     [rd.random(), rd.random(), rd.random()],
#     [rd.random(), rd.random(), rd.random()]
# ])

for i in range(100000):
    y = feed_forward(w, x)
    w = back_propagation(y_out, y, w, x)

print(w)

print("-------")
print("0 1 1")
final_problem = feed_forward(w, [0, 1, 1])
print(final_problem)

result = 1 if final_problem > 0.5 else 0
print(result)


print("-------")
print("0 1 0")
final_problem = feed_forward(w, [0, 1, 0])
print(final_problem)

result = 1 if final_problem > 0.5 else 0
print(result)


print("-------")
print("1 0 1")
final_problem = feed_forward(w, [1, 0, 1])
print(final_problem)

result = 1 if final_problem > 0.5 else 0
print(result)