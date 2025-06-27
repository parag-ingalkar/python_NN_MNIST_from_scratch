from neural_network.model.neural_net import NeuralNetwork
from neural_network.layers import Linear, ReLU
from neural_network.loss.cross_entropy import SoftmaxCrossEntropy
from neural_network.train.trainer import train
from neural_network.utils.data_loader import load_mnist_data

def main():

    X_train, Y_train, X_test, Y_test = load_mnist_data("mnist.config")

    model = NeuralNetwork([
        Linear(784, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, 10)
    ])

    loss_fn = SoftmaxCrossEntropy()

    print('\nTraining started...\n')
    train(model, X_train, Y_train, loss_fn, epochs = 10, batch_size = 64, learning_rate = 0.01)

    print('\nEvaluating Model...\n')
    accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Accuracy = {accuracy}\n")



if __name__ == "__main__":
    main()
