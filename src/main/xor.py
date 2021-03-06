import numpy as np
import plotly.express as px
import pandas as pd

# https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets
inputs = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
)
expected_output = np.array(
    [
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 1]
    ]
)

epochs = 100000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 2

# Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))

accuracy = [[],[]]

# Training algorithm
for count in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    accuracy[0].append(count)
    accuracy[1].append(-1 * error[0, 0])

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr


row = 3
print("")
hid1 = hidden_weights[0, 0] * inputs[row, 0] + hidden_weights[1, 0] * inputs[row, 1]
sum1 = hid1+hidden_bias[0, 0]
sigm1 = sigmoid(sum1)
print(str(inputs[row, 0]) + " -- " + str(hidden_weights[0, 0]) + " -- wx=" + str(hid1))
print(str(inputs[row, 1]) + " -- " + str(hidden_weights[1, 0]) + " -- ||| "
      + " | b=" + str(hidden_bias[0, 0]) + " ||| sum=" + str(sum1)
      + " -> sigm=" + str(sigm1) + " (H1)")
print("")
hid2 = hidden_weights[0, 1] * inputs[row, 0] + hidden_weights[1, 1] * inputs[row, 1]
sum2 = hid2+hidden_bias[0, 1]
sigm2 = sigmoid(2)
print(str(inputs[row, 0]) + " -- " + str(hidden_weights[0, 1]) + " -- wx=" + str(hid2))
print(str(inputs[row, 1]) + " -- " + str(hidden_weights[1, 1]) + " -- ||| "
      + " | b=" + str(hidden_bias[0, 1]) + " ||| sum=" + str(sum2)
      + " -> sigm=" + str(sigm2) + " (H2)")

print("")
hid3 = output_weights[0, 0] * sigm1 + output_weights[1, 0] * sigm2
sum3 = hid3+output_bias[0, 0]
sigm3 = sigmoid(sum3)
print("(H1)-" + str(sigm1) + " -- " + str(output_weights[0, 0]) + " -- wx=" + str(hid3))
print("(H2)-" + str(sigm2) + " -- " + str(output_weights[1, 0]) + " -- ||| "
      + " | b=" + str(output_bias[0, 0]) + " ||| sum=" + str(sum3)
      + " -> sigm=" + str(sigm3))



#
# print("Final hidden weights: ", end='')
# print(*hidden_weights)
# print("Final hidden bias: ", end='')
# print(*hidden_bias)
# print("Final output weights: ", end='')
# print(*output_weights)
# print("Final output bias: ", end='')
# print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ", end='')
print(*predicted_output)

# fig = px.line(x=accuracy[0], y=accuracy[1])
# fig.show()
