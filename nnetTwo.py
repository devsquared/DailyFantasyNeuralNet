import numpy as np
import scipy.special

class LayerOfNeurons():
    def __init__(self, number_of_inputs, number_neurons):
        self.weights = 2 * np.random.random((number_of_inputs, number_neurons)) - 1
        self.bias = np.zeros((1, number_neurons))

class NNet():
    def __init__(self, hidden_layer, output_layer,):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.output_error_array = []
        self.val_output_error_array = []
        self.mse_output_error_array = []

    #Sigmoid function. Set up for activation function to normalize between 0 to 1.
    def __sigmoid(self, x):
        return scipy.special.expit(x)

    #Sigmoid derivative. Tells us gradient and confidence about weight.
    #Used in back prop and honing in weights.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    #X is input
    def __forward_prop(self, X):
        z1 = X.dot(self.hidden_layer.weights) + self.hidden_layer.bias
        a1 = self.__sigmoid(z1)
        z2 = a1.dot(self.output_layer.weights) + self.output_layer.bias
        a2 = self.__sigmoid(z2)
        return a1, a2

    def train(self, training_set_inputs, training_labels, val_set_inputs, val_labels, learning_rate=0.1, iterations=10000):
        for iter in xrange(iterations):
            a1, a2 = self.__forward_prop(training_set_inputs)
            va1, va2 = self.__forward_prop(val_set_inputs)
            output_error = training_labels - a2
            val_ouput_error = val_labels - va2

            if (iter% 1) == 0:
                self.output_error_array.append(np.mean(np.abs(output_error)))
                self.mse_output_error_array.append(np.square(np.mean(output_error)))
                self.val_output_error_array.append(np.mean(np.abs(val_ouput_error)))

            output_delta = output_error * self.__sigmoid_derivative(a2)

            hidden_error = output_delta.dot(self.output_layer.weights.T)
            hidden_delta = hidden_error * self.__sigmoid_derivative(a1)

            hidden_adjustment = training_set_inputs.T.dot(hidden_delta)
            output_adjustment = a1.T.dot(output_delta)

            self.hidden_layer.weights += hidden_adjustment * -learning_rate
            self.output_layer.weights += output_adjustment * -learning_rate
            self.hidden_layer.bias += np.sum(hidden_delta, axis=0, keepdims=True) * -learning_rate
            self.output_layer.bias += np.sum(output_delta, axis=0) * -learning_rate

    def predict(self, X):
        return self.__forward_prop(X)

