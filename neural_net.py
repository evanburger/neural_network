import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Neural_Net(object):
    def __init__(self):
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1
        self.weights_1_size = lambda _:(self.input_size, self.hidden_size)
        self.weights_2_size = lambda _:(self.hidden_size, self.output_size)
        self.input_layer = None
        self.hidden_layer = None
        self.output_layer = None
        self.weights_1 = np.random(weights_1_size)
        self.weights_2 = np.random(weights_2_size)

    def feed_forward(self):
        pass

    def back_propagate(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass
