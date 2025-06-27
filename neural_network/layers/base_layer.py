class Layer:
    def forward(self, input):
        raise NotImplementedError("forward() not implemented")

    def backward(self, grad_output, learning_rate):
        raise NotImplementedError("backward() not implemented")
