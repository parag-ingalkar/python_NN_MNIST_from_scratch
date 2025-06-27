import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)
    
    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def evaluate(self, x, y):
        logits = self.forward(x)
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(preds == labels)
