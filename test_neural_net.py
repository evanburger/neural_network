import pytest
import numpy as np

from neural_net import Neural_Network as NN


def test_activate():
    z_0 = 0.
    result_0 = NN._activate(z_0)
    assert result_0 == 0.5

    z = -42.
    result_42 = NN._activate(z)
    assert (result_42 <= 1.) and (result_42 >= 0.)
    
    assert isinstance(result_0, float) or isinstance(result_42, float)

def test_predict():
    input_vector = np.array([0.5, 0.5])
    output_vector = NN._predict(NN, input_vector)
    
    assert isinstance(output_vector, np.array)
    
    for item in ouput_vector:
        assert isinstance(item, float)
        assert (item <= 1.) and (item >= 0.)

def test_loss():
    predicted_vector, target_vector = np.array([0.5, 0.5]), np.array([0.5, 0.5])
    model_loss = NN._loss(NN, predicted_vector, target_vector)
    
    assert isinstance(model_loss, float)
    assert model_loss == 0.

def test_activate_prime():
    z_0, z_1, z_neg1 = 0., 0.1, -0.1
    result_0 = NN._activate_prime(z_0)
    result_1 = NN._activate_prime(z_1)
    result_neg1 = NN._activate_prime(z_neg1)
    assert isinstance(result_0, float)
    assert (result_0 > result_1) and (result_0 > result_neg1)
