# Python Neural Network MNIST from Scratch

A simple implementation of a fully connected neural network in Python and numpy from scratch to classify handwritten digits from the MNIST dataset—without using any deep learning frameworks.

## Overview

This repository includes a package - neural_network with implementation of Neural Network architecture using numpy. It walks through building a neural network from the ground up.

Data loading: Raw MNIST format loading and preprocessing.

Network architecture: Configurable input, hidden, and output layers.

Forward propagation: Compute activations using customizable activation functions.

Backward propagation: Manual gradient calculation to update weights and biases.

Training loop: Train with mini-batches, track losses and accuracy metrics.

Evaluation: Validate model performance on test data.

Great for educational purposes or anyone wanting insight into the nuts and bolts of neural networks.

The repository also includes 3 jupyter notebooks each dedicated to solving the MNIST dataset using different frameworks.

### MNIST_Neural_Network.ipynb

    - Includes bare bone implementation of neural network architecture using only numpy and idx2numpy.
    - Logically similar to the package neural_network but without OOP principle.
    - Difficult to scale, but great for learning the Neural Netowrk learning process,

### MNIST_TensorFlow.ipynb

    - Includes the implentation of a simple Neural network for MNIST data set hand written digit recognition using TensorFlow.
    - A very well structured framework, with minimum setup required.
    - Probably the easiest way to simply set up a Neural Network fast and efficiently.

### MNIST_PyTorch.ipynb

    - Includes the same Neural Network architecture using PyTorch.
    - The library requires a few lines of code to setup your model and train it.
    - It provides somewhat a middle ground between, TensorFlow's high level implementation and the low level implementation of our custom implemnetation.

## Features

- Trains a fully connected network (e.g., input → one/two hidden layers → softmax output).
- Uses ReLU / SoftMax activations and Cross-Entropy Loss.
- Mini-batch stochastic gradient descent training pipeline.
- Monitors training and test accuracy throughout epochs.
- Easily configurable hyperparameters: layer sizes, learning rate, batch size, epochs.

## Installation

Requires Python 3.12 and above.
Note: Currently TensorFlow is not available for Python 3.13

Notebook requires below packages
tensorflow==2.19.0
torch==2.7.1
torchvision==0.22.1
matplotlib==3.10.3

The project uses uv for managing dependencies.
All the required depencies can be found in uv.lock file.

Installation procedure

1. Install uv if not installed already

```bash
# Linux / MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternatively use pip or pipx
pip install uv
pipx install uv
```

Visit https://docs.astral.sh/uv/getting-started/installation/ for more details.

2. Clone the repository

```bash
git clone https://github.com/parag-ingalkar/python_NN_MNIST_from_scratch.git
cd python_NN_MNIST_from_scratch
```

3. Sync with uv.lock to install requirements

```bash
uv sync
```

## Usage

Run the training script with default configurations:

```bash
uv run main.py
```

Edit main.py to tweak hyperparameters:

Layers: e.g.
model = NeuralNetwork([
Linear(784, 128),
ReLU(),
Linear(128, 10)
])

learning_rate: e.g. 0.01

batch_size: e.g. 64

epochs: e.g. 20

## Results

Once training finishes, you'll see printed logs like:

Training started...

Epoch 1, Loss: 0.0146, Accuracy: 0.8714
...
Epoch 10, Loss: 0.0028, Accuracy: 0.9511

Evaluating Model...

Test Accuracy = 0.947

## Project Structure

```
python_NN_MNIST_from_scratch
├──MNIST_data                       # MNIST data files
│   ├──t10k-images-idx3-ubyte
│   ├──t10k-labels-idx1-ubyte
│   ├──train-images-idx3-ubyte
│   └──train-labels-idx1-ubyte
├──neural_network                   # Package
│   ├──layers
│   │   ├──__init__.py
│   │   ├──base_layer.py
│   │   ├──linear.py                # Dense / Fully connected layer
│   │   └──relu.py                  # ReLU activation
│   ├──loss
│   │   ├──__init__.py
│   │   └──cross_entropy.py         # Categorical cross entropy loss function
│   ├──model
│   │   ├──__init__.py
│   │   └──neural_net.py            # Model Class
│   ├──train
│   │   ├──__init__.py
│   │   └──trainer.py               # Includes Train function for better abstraction
│   ├──utils
│   │   ├──__init__.py
│   │   ├──config_loader.py         # Load your config file
│   │   └──data_loader.py           # Includes mnist_data_loader function
│   └──__init__.py
├──main.py                          # Implement your architecture
├──MNIST_Neural_Network.ipynb       # Implementation in numpy
├──MNIST_PyTorch.ipynb              # Implementation in PyTorch
├──MNIST_TensorFlow.ipynb           # Implementation in TensorFlow
├──mnist.config                     # Contains path to data files
├──pyproject.toml                   # Project settings with dependencies
├──README.md
├──uv.lock                          # Lock file with exact versions for reproduce environment
├──.gitignore
└──.python-version
```

## How It Works

load_mnist_data : Parses the config file and loads the train and test data.

NeuralNetwork class: Defines and Initializes the layers to be used in the architecture.

Layer class: Linear - Holds weights, biases and defines forward/backward propagation and updates.
ReLU - Defines forward and backward propagation function for ReLU.

Training loop:

    1. Sample batches.
    2. Forward-propagate inputs.
    3. Compute loss.
    4. Backward-propagate gradients.
    5. Update parameters.

Testing model:

    1. Forward propagte inputs.
    2. Get predictions from output.
    3. Calculate accuracy w.r.t. ground truth labels.

## Further Improvements

- Add learning rate decay or momentum optimizers.
- Support additional modules (e.g. dropout, batch normalization).
- Support additional activation functions.
- Implement convolutional layers.
- Visualize losses and metrics with graphical libraries.

## Contributing

Contributions welcome! Feel free to open issues or create pull requests for bug fixes, refactoring, or new features.

## License

MIT License — feel free to use, adapt, and distribute!
