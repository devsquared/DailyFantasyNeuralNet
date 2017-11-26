from numpy import exp, array, random, dot

"""
Neural network with multi layers. Simple approach. Built with 1 hidden layer. 
"""

class LayerOfNeurons():
    def __init__(self, number_neurons, number_of_inputs):
        self.weights = 2 * random.random((number_of_inputs, number_neurons)) - 1

class NNet():
    def __init__(self, hidden_layer, output_layer):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

    #If need be, set up different activation functions and their derivatives.

    #Sigmoid function. Set up for activation function to normalize between 0 to 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    #Sigmoid derivative. Tells us gradient and confidence about weight.
    #Used in back prop and honing in weights.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    #Through a process of trial and error, train the net. Adjusts weights each pass.
    def train(self, training_set_in, training_set_labels, iterations):
        for iter in xrange(iterations):
            #pass training into net
            output_of_hidden_layer, output_of_output = self.eval(training_set_in)

            #Back Prop to asses weights.

            #Calc error for output. 
            #Defined as difference of training labels and predicted output.
            output_error = training_set_labels - output_of_output
            output_delta = output_error * self.__sigmoid_derivative(output_of_output)

            #Calc error for hidden layer.
            #Defined as the contribution to the error in output.
            hidden_error = output_delta.dot(self.output_layer.weights.T)
            hidden_delta = hidden_error * self.__sigmoid_derivative(output_of_hidden_layer)

            #Adjust weights according to error found.

            #calculate adjustments.
            hidden_adjustment = training_set_in.T.dot(hidden_delta)
            output_adjustment = output_of_hidden_layer.T.dot(output_delta)

            #Adjust the weights.
            self.hidden_layer.weights += hidden_adjustment
            self.output_layer.weights += output_adjustment

    #Evaluate the nerual net. 
    def eval(self, inputs):
        output_of_hidden_layer = self.__sigmoid(dot(inputs, self.hidden_layer.weights))
        output_of_output = self.__sigmoid(dot(output_of_hidden_layer, self.output_layer.weights))
        return output_of_hidden_layer, output_of_output

    #Printing weights for manual eval.
    def print_weights(self):
        print "Hidden Layer weights: "
        print self.hidden_layer.weights
        print "Output Layer weights: "
        print self.output_layer.weights

