"""This module contains the neural net's class."""
import numpy as np


RELU_ALPHA = 0.1
GRAD_LOWER_LIM = -1000
GRAD_UPPER_LIM = 1000


class Neural_Network(object):
    """An artificial neural network written from scratch.
    
    It has only 1 hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        # These are the hyperparameters.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # Randomly set the parameters.
        self.randomize()

    @staticmethod
    def _he_var(fan_in):
        return np.sqrt(2/fan_in)

    @staticmethod
    def _kaiming_var(fan_in):
        return np.sqrt(2/fan_in)

    @staticmethod
    def _sigmoid(z):
        # Return a float calculated using the sigmoid function given a float.
        # Formula: f(z) = 1 / (1 + e^(-z))
        return 1 / (1+np.exp(-z))

    @staticmethod
    def _relu(z):
        # Return a float calculated using the sigmoid function given a float.
        # This uses the leaky ReLU.
        # Formula: f(z) = max(0, z)
        return np.where(z>0, z, z*RELU_ALPHA)

    @staticmethod
    def _relu_prime(z):
        # Return a float calculated using the sigmoid function given a float.
        # This uses the leaky ReLU.
        z[z>0] = 1
        z[z<=0] = RELU_ALPHA
        return z

    @staticmethod
    def _sigmoid_prime(z):
        # Return a float calculated using the derivative of the sigmoid function given a float.
        # Formula: f'(z) = e^(-z) / (1 + e^(-z))^2
        return (np.exp(-z)) / (1+np.exp(-z))**2

    @staticmethod
    def _convert_10D_to_int(vector_10D):
        return list(vector_10D).index(max(vector_10D))

    def _scale_weights(self, fan_in):
        if self.activation == "sigmoid":
            return self._kaiming_var(fan_in)
        elif self.activation == "relu":
            return self._he_var(fan_in)
        else:
            return ValueError

    def _activate(self, z):
        # Return a float calculated using the function determined by self.activation given a float.
        if self.activation == "sigmoid":
            return self._sigmoid(z)
        elif self.activation == "relu":
            return self._relu(z)
        else:
            return ValueError

    def _activate_prime(self, z):
        # Return a float calculated using the function determined by self.activation given a float.
        if self.activation == "sigmoid":
            return self._sigmoid_prime(z)
        elif self.activation == "relu":
            return self._relu_prime(z)
        else:
            return ValueError

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
        if self.activation == "sigmoid":
            # The gradient should be clipped to avoid floating point overflow errors.
            self.z2 = np.clip(self.z2, GRAD_LOWER_LIM, GRAD_UPPER_LIM)
        delta2 = -(target_vector-predicted_vector) * self._activate_prime(self.z2)
        # Some transposes of their respective matrices must be used.
        dJdW2 = np.dot(self.a.T, delta2)
        if self.activation == "sigmoid":
            # The gradient should be clipped to avoid floating point overflow errors.
            self.z1 = np.clip(self.z1, GRAD_LOWER_LIM, GRAD_UPPER_LIM)
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
        """Return a vector as the prediction of the given input vector."""
        # Formula: y_hat = f(X W_1) W_2
        self.z1 = np.dot(input_vector, self.W1)
        self.a = self._activate(self.z1)
        self.z2 = np.dot(self.a, self.W2)
        output_vector = self._activate(self.z2)
        return output_vector

    def train(self, training_input, target_vector, validation_size=None, batch_size=None, learning_rate=1e-1, epochs=1, verbose=False,):
        """Iterate through epochs number of times, updating the model each time.
        An int for epochs (default is 1), an np.array for training_input, an np.array for target_vector,
        and a float for learning_rate (default is 0.1), an int for validation_size
        and an int for the batch_size must be given. If verbose is True,
        print out the current epoch and loss.""" 
        if validation_size is not None:
            validation_examples, training_input = training_input[:validation_size], training_input[validation_size:]
            validation_labels, target_vector = target_vector[:validation_size], target_vector[validation_size:]
            validation_results = [self.test(validation_examples, validation_labels)]

        if batch_size is None:
            batch_size = len(training_input)
            iterations = 1
        iterations = int(len(training_input) / batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            for iteration in range(iterations):
                batch_x = training_input[iteration*batch_size:(iteration+1)*batch_size]
                batch_y = target_vector[iteration*batch_size:(iteration+1)*batch_size]
                model_loss = self._update_model(batch_x, batch_y, learning_rate) / len(batch_x) * 2
                if verbose:
                    print(f"{iteration}: {model_loss}")
            if validation_size is not None:
                testing_result = self.test(validation_examples, validation_labels)
                validation_results.append(testing_result)
                # The model should stop training if the last test is worse than the previous one to prevent overfitting.
                if validation_results[-1] < validation_results[-2]:
                    return validation_results
            else:
                testing_result = self.test(training_input, target_vector)
                return testing_result
            if verbose:
                if "validation_results" in locals():
                    print(f"Accuracy history: {validation_results}")
                else:
                    print(f"Accuracy: {testing_result}")
                    return testing_result

    def randomize(self):
        """Set weights to a random state."""
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * self._scale_weights(self.input_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * self._scale_weights(self.hidden_size)

    def test(self, testing_input, target_vector):
        """Return a float for the error of the network given a matrix
        for the testing_input and a vector for the target_vector."""
        accuracy = 0
        amount = len(testing_input)
        for i in range(amount):
            y_hat = self._convert_10D_to_int(self.predict(testing_input[i]))
            y = self._convert_10D_to_int(target_vector[i])
            if y_hat == y:
                accuracy += 1
        return accuracy / amount

    def set_weights(self, weights_tuple):
        """Set the network's weights given weights as a tuple."""
        self.W1 = weights_tuple[0].copy()
        self.W2 = weights_tuple[1].copy()
