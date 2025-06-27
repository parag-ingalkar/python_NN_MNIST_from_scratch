import numpy as np
from .base_layer import Layer

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output, learning_rate):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input
