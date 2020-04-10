"""This module contains the neural net's class."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


INPUT_SIZE = 2
HIDDEN_SIZE = 2
OUTPUT_SIZE = 1


class Neural_Network(object):
    """An artificial neural network written from scratch to predict XOR logic."""
    def __init__(self, input_size, hidden_size, output_size):
        # These are the hyperparameters.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # These are the parameters.
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    @staticmethod
    def _activate(z):
        # Return a float calculated using the sigmoid function given a float.
        # Formula: f(z) = 1 / (1 + e^(-z))
        return 1 / (1+np.exp(-z))

    @staticmethod
    def _activate_prime(z):
        # Return a float calculated using the derivative of the sigmoid function given a float.
        # Formula: f'(z) = e^(-z) / (1 + e^(-z))^2
        return (np.exp(-z)) / (1+np.exp(-z))**2

    def _predict(self, input_vector):
        # Return a vector as prediction of the given input vector.
        # Formula: y_hat = f(X W_1) W_2
        self.z1 = np.dot(input_vector, self.W1)
        self.a = self._activate(self.z1)
        self.z2 = np.dot(self.a, self.W2)
        output_vector = self._activate(self.z2)
        return output_vector

    def _loss(self, predicted_vector, target_vector):
        # Return a float for the loss of the model given a predicted vector and a target vector.
        # Formula: J = SIGMA 1/2 (y - y_hat)^2
        model_loss = np.sum(1/2 * (predicted_vector - target_vector)**2)
        return model_loss


if __name__ == "__main__":
    nn = Neural_Network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
