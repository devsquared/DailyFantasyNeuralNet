import nnetTwo
from numpy import exp, array, random, arange, mean
import pandas
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #reading in the data
    df = pandas.read_csv('DatasetQB.csv')

    columns_to_drop_for_hard_stats = ['Player', 'Pos', 'Week', 'Team', 'Opp', 'Comp', 'PAtt', 'Pct', 'PYds', 'PYds/Att', 'Pass_TD', 'QB_Rating', 'RAtt', 'RYds', 'RYds/Att', 'Rush_TD', 'Fantasy_Points', 'Opp_Rank', 'Opp_Pos_Rank', 'Salary', 'Projection', 'Avg_Salary', 'SalaryDiffToAvg', 'Preseason Proj./game', 'Label', 'LabelRef']
    columns_to_drop_for_fant_stats = ['Player', 'Pos', 'Week', 'Team', 'Opp', 'Comp', 'PAtt', 'Pct', 'NPct', 'PYds', 'NPYds', 'PYds/Att', 'Pass_TD', 'Int', 'QB_Rating', 'NQB_Rating', 'RAtt', 'RYds', 'RYds/Att', 'Rush_TD', 'Total_TD', 'Fantasy_Points', 'Salary', 'Avg_Salary', 'Label', 'LabelRef']

    #split dataset into training, validation, and testing sets (70%, 20%, 10% resp.)
    training_hard = df.sample(frac = 0.7)
    df.drop(training_hard.index)
    validation_hard = df.sample(frac = .2)
    df.drop(validation_hard.index)
    test_hard = df.sample(frac = .1)

    training_set_in_hard = training_hard.drop(columns_to_drop_for_hard_stats, 1)
    validation_set_in_hard = validation_hard.drop(columns_to_drop_for_hard_stats, 1)

    training_set_in_list_hard = array(training_set_in_hard.values.tolist())
    validation_set_in_list_hard = array(validation_set_in_hard.values.tolist())

    # for now, labels are defined as doing better than projected (will build multiple options for position)
    training_labels = array([training_hard['LabelRef'].tolist()]).T
    validation_labels = array([validation_hard['LabelRef'].tolist()]).T

    epochs = 500

    hidden_layer = nnetTwo.LayerOfNeurons(5, 8)
    output_layer = nnetTwo.LayerOfNeurons(8, 1)

    net_hard = nnetTwo.NNet(hidden_layer, output_layer)

    net_hard.train(training_set_in_list_hard, training_labels, validation_set_in_list_hard, validation_labels, learning_rate=0.001, iterations=epochs)

    #predict some values for week 13 nfl, compare to tables

    #move to exp 2 and 3
    # replot learning curve and fix x label


        """
        Below Used for plotting learning curve
        ......
        epochs_checks = epochs/1
        x = arange(start=0, stop=epochs_checks, step=1)
        plt.figure()
        training, = plt.plot(x, error, color='r', label="training_data")
        validation, = plt.plot(x, val_error, color='b', label="validation")
        plt.ylim(0, 1)
        plt.xlabel('epoch checkpoints (epochs/1000)')
        plt.ylabel('MSE')
        plt.title('Learning Curve: %s' % lr)
        plt.legend([training, validation], ['Training', 'Validation'])
        plt.savefig("LearningCurveWith%sNeuronsLR%s.png" % (dim, lr))
        plt.close()
        """

# find optimal values from above, then check mse to find best amount of epochs
