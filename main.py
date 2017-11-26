import nnet
from numpy import exp, array, random

if __name__ == "__main__":

    random.seed(1)

    hidden_layer = nnet.LayerOfNeurons(4, 3)

    output_layer = nnet.LayerOfNeurons(1, 4)

    net = nnet.NNet(hidden_layer, output_layer)

    print "First stage: Random weights:"
    net.print_weights()

    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    net.train(training_set_inputs, training_set_outputs, 60000)

    print "Second stage: Weights after training:"
    net.print_weights()

    print "Final stage: prediction:"
    hidden_state, output = net.eval(array([1,1,0]))
    print output