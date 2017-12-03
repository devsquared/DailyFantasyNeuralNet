import nnet
from numpy import exp, array, random
import pandas

if __name__ == "__main__":
    
    #reading in the data
    df = pandas.read_csv('Dataset.csv')

    columns = ['PLAYER', 'PSP', 'AWP', 'LBL']

    #split dataset into training, validation, and testing sets (70%, 20%, 10% resp.)
    training = df.sample(frac = 0.7)
    df.drop(training.index)
    validation = df.sample(frac = .2)
    df.drop(validation.index)
    test = df.sample(frac = .1)

    training_set_in = training.drop(columns, 1)
    validation_set_in = validation.drop(columns, 1)

    training_set_in_list = training_set_in.values.tolist()
    validation_set_in_list = validation_set_in.values.tolist()

    # for now, labels are defined as doing better than projected (will build multiple options for position)
    training_labels = array([training['LBL']]).T
    validation_labels = array([validation['LBL']]).T

    print str(training_labels) + '\n\n'
    print str(validation_labels)

    hidden_layer = nnet.LayerOfNeurons(4,3)

    output_layer = nnet.LayerOfNeurons(1,4)

    net = nnet.NNet(hidden_layer, output_layer)

    net.train(training_set_in_list, training_labels, validation_set_in_list, validation_labels, error_target=0.5, max_iterations=1000)

    #test on testing set
    for row in test.itertuples():
        print getattr(row, "PLAYER") + ' : '
        input_array = [getattr(row, "PPPPG"), getattr(row, "WPP"), getattr(row, "PWAP")]
        hidden_state, output = net.eval(array())
        print output + '\n'

# First Exp: 
#           inputs: weekly proj points
#                   prior week performance
#                   preseason point projection

# Second Exp:
#           inputs: same as one
#           + salary / 1000

# maybe look at throwing out fantasy stats and using hard stats only to see
# if that predicts differently
# look into profootballreference and get the csv's that i need

# Third Exp:
#           inputs: same as two
#           + games played this year