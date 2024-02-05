import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./archive/mnist_test.csv')
data = np.array(data)
m, n = data.shape


train_data = data.T
label_data = train_data[0]
pixel_data = train_data[1: n]
normalised_pixel_data = pixel_data / 255
_, m_dimension = normalised_pixel_data.shape

def init_data():
    first_random_weights = np.random.rand(10, 784) - 0.5
    first_random_bias = np.random.rand(10, 1) - 0.5
    second_random_weights = np.random.rand(10, 10) - 0.5
    second_random_bias = np.random.rand(10, 1) - 0.5
    return first_random_weights, first_random_bias, second_random_weights, second_random_bias

def init_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2


def ReLU(X):
    return np.maximum(X, 0)

def SoftMax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_propogation(
    first_layer_weights,
    first_layer_bias,
    second_layer_weights,
    second_layer_bias,
    pixel_data
    ) :
    unactivated_first_layer = first_layer_weights.dot(pixel_data) + first_layer_bias
    activated_first_layer = ReLU(unactivated_first_layer)
    unactivated_second_layer = second_layer_weights.dot(activated_first_layer) + second_layer_bias
    activated_second_layer = SoftMax(unactivated_second_layer)
    return unactivated_first_layer, activated_first_layer, unactivated_second_layer, activated_second_layer

def one_hot_matrix_generator(Y):
    matrix = np.zeros((Y.size, Y.max() + 1))
    matrix[np.arange(Y.size), Y] = 1 
    return matrix.T

def derivative_RuLU(X):
    return X > 0

def backwards_propogation(
    unactivated_first_layer,
    activated_first_layer,
    activated_second_layer,
    second_layer_weights,
    label_data,
    pixel_data,
    ) :
    one_hot_matrix = one_hot_matrix_generator(label_data)
    inverse_dimension = 1 / m
    difference_in_second_layer = activated_second_layer - one_hot_matrix
    difference_to_second_layer_weights = inverse_dimension * difference_in_second_layer.dot(activated_first_layer.T)
    difference_to_second_layer_bias = inverse_dimension * np.sum(difference_in_second_layer)
    difference_in_first_layer = second_layer_weights.T.dot(difference_in_second_layer) * derivative_RuLU(unactivated_first_layer)
    difference_to_first_layer_weights = inverse_dimension * difference_in_first_layer.dot(pixel_data.T)
    # print(difference_in_first_layer, difference_to_first_layer_weights)
    difference_to_first_layer_bias = inverse_dimension * np.sum(difference_in_first_layer)
    return (
        difference_to_first_layer_weights, 
        difference_to_first_layer_bias, 
        difference_to_second_layer_weights, 
        difference_to_second_layer_bias
    )

def update_parameters(
    difference_to_first_layer_weights, 
    difference_to_first_layer_bias, 
    difference_to_second_layer_weights, 
    difference_to_second_layer_bias,
    first_layer_weights,
    first_layer_bias,
    second_layer_weights,
    second_layer_bias,
    learning_pace,
    ) :
    new_first_layer_weights = first_layer_weights - learning_pace * difference_to_first_layer_weights
    new_first_layer_bias = first_layer_bias - learning_pace * difference_to_first_layer_bias
    new_second_layer_weights = second_layer_weights - learning_pace * difference_to_second_layer_weights
    new_second_layer_bias = second_layer_bias - learning_pace * difference_to_second_layer_bias
    return (
        new_first_layer_weights,
        new_first_layer_bias,
        new_second_layer_weights,
        new_second_layer_bias,
    )
    
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
    
def gradient_descent(pixel_data, label_data, learning_pace, iterations):
    first_weights, first_bias, second_weights, second_bias = init_data()
    for i in range(iterations) :
        ufl, afl, usl, asl = forward_propogation(first_weights, first_bias, second_weights, second_bias, pixel_data)
        dflw, dflb, dslw, dslb = backwards_propogation(ufl, afl, asl, second_weights, label_data, pixel_data)
        first_weights, first_bias, second_weights, second_bias = update_parameters(dflw, dflb, dslw, dslb, first_weights, first_bias, second_weights, second_bias, learning_pace)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(asl)
            print(get_accuracy(predictions, label_data))
    return first_weights, first_bias, second_weights, second_bias
            
W1, b1, W2, b2 = gradient_descent(normalised_pixel_data, label_data, 0.10, 10000)
# print(activated_second_layer)

# print(init_data())