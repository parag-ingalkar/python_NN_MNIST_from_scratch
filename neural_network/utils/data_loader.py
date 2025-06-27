import numpy as np
import idx2numpy
import os
from .config_loader import load_config

def resolve_path(base_path, relative_path):
    return os.path.join(os.path.dirname(base_path), relative_path)

def one_hot_encode(labels):
    labels = labels.flatten()
    num_classes = np.max(labels) + 1
    one_hot = np.eye(num_classes)[labels]
    return one_hot

def load_mnist_data(config_path):
    config = load_config(config_path)

    # Paths from config
    train_images_path = resolve_path(config_path, config['rel_path_train_images'])
    train_labels_path = resolve_path(config_path, config['rel_path_train_labels'])
    test_images_path  = resolve_path(config_path, config['rel_path_test_images'])
    test_labels_path  = resolve_path(config_path, config['rel_path_test_labels'])

    # Load and preprocess training data
    X_train = idx2numpy.convert_from_file(train_images_path)
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0

    Y_train = idx2numpy.convert_from_file(train_labels_path)
    Y_train = one_hot_encode(Y_train)

    # Load and preprocess test data
    X_test = idx2numpy.convert_from_file(test_images_path)
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

    Y_test = idx2numpy.convert_from_file(test_labels_path)
    Y_test = one_hot_encode(Y_test)

    return X_train, Y_train, X_test, Y_test
