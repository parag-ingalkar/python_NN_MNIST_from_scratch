import numpy as np
from .base_layer import Layer

class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.biases = np.zeros((1, out_features))
    
    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output) / self.input.shape[0]
        grad_biases = np.sum(grad_output, axis=0, keepdims=True) / self.input.shape[0]

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_input


