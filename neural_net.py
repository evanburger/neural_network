"""This module contains the neural net's class."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Neural_Network(object):
    """An artificial neural network written from scratch.
    
    It has only 1 hidden layer.
    """
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

    def _loss(self, predicted_vector, target_vector):
        # Return a float for the loss of the model given a predicted vector and a target vector.
        # Formula: J = SIGMA 1/2 (y - y_hat)^2
        model_loss = np.sum(1/2 * (predicted_vector - target_vector)**2)
        return model_loss

    def _loss_prime(self, input_vector, target_vector):
        # Return a dictionary of np.array for each set of weights given an input vector
        # and target vector.
        # The gradient points "uphill" in the loss function space.
        # Formula: DELTA_2 = -(y-y_hat) * f'(z_2)
        #          dJ/dW_2 = a.T DELTA_2
        #          DELTA_1 = f'(z_1) * (DELTA_2 W_2.T)
        #          dJ/dW_1 = X.T DELTA_1
        
        # The variables must be populated.
        predicted_vector = self.predict(input_vector)
        delta2 = -(target_vector-predicted_vector) * self._activate_prime(self.z2)
        # Some transposes of their respective matrices must be used.
        dJdW2 = np.dot(self.a.T, delta2)
        delta1 = self._activate_prime(self.z1 ) * np.dot(delta2, self.W2.T)
        dJdW1 = np.dot(input_vector.T, delta1)
        gradient = {"dJdW2": dJdW2, "dJdW1": dJdW1}
        return gradient

    def _update_model(self, input_matrix, target_vector, learning_rate):
        # Return a float for the loss of the model given an input matrix,
        # target vector and a float for the learning rate.
        gradient = self._loss_prime(input_matrix, target_vector)
        self.W1 -= learning_rate*gradient["dJdW1"]
        self.W2 -= learning_rate*gradient["dJdW2"]
        predicted_vector = self.predict(input_matrix)
        model_loss = self._loss(predicted_vector, target_vector)
        return model_loss

    def predict(self, input_vector):
        # Return a vector as prediction of the given input vector.
        # Formula: y_hat = f(X W_1) W_2
        self.z1 = np.dot(input_vector, self.W1)
        self.a = self._activate(self.z1)
        self.z2 = np.dot(self.a, self.W2)
        output_vector = self._activate(self.z2)
        return output_vector

    def train(self, training_input, target_vector, learning_rate, epoch_size=100, verbose=False):
        """Iterate through epoch_size number of times, updating the model each time.
        An int for epoch_size (default is 10), an np.array for training_input, an np.array for target_vector,
        and a float for learning_rate must be given. If verbose is True,
        print out the current epoch and loss."""
        for epoch in range(epoch_size):
            model_loss = self._update_model(training_input, target_vector, learning_rate)
            if verbose:
                print("{epoch}: {loss}".format(epoch=epoch, loss=model_loss))

    def randomize_parameters(self):
        """Set weights to a random state."""
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)        
