import argparse
import pickle
import sys
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from model import Model


def plot_scatter(df):
    ax = df.plot.scatter(x='population',
                         y='profit',
                         c='Red',
                         marker='x')
    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel("Profit in $10,000s")
    plt.savefig('figs/scatter_plot_training_data.png')


def plot_data_with_line(df):
    # check if there is a file called univariate.pkl in model/
    file_list = [f for f in listdir('model/') if isfile(join('model/', f))]
    if 'univariate.pkl' in file_list:
        model = Model(vector_size=df.shape[1])
        model.load_from_file('model/univariate.pkl')
        weights = model.theta
        slope = weights[1][0]  # theta_1
        intercept = weights[0][0]  # theta_0
        print('Slope= ', slope)
        print('Intercept= ', intercept)

        # plot the training data directly from the dataframe
        ax = df.plot.scatter(x='population',
                             y='profit',
                             c='Red',
                             marker='x',
                             label='Training data')

        # define the data for plotting the regression line
        x_list = range(4, 24)
        y_list = intercept + slope * x_list
        ax.plot(x_list, y_list, '-b', label='Linear regression')

        # define labels for x- and y- axis
        ax.set_xlabel('Population of City in 10,000s')
        ax.set_ylabel("Profit in $10,000s")

        plt.legend()
        plt.savefig('figs/data_with_line.png')
    else:
        sys.exit('There are no model weights. Unable to plot the regression line!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # define the arguments that may be given as an input to the app
    # the app will have a positional argument which is 'type', a required argument 'action' and an optional argument 'plot_type'
    parser.add_argument("type",
                        help='Regression type', choices=['univariate', 'multivariate'])
    parser.add_argument("--action", required=True, help="determines the functionality to be executed",
                        choices=['train', 'predict', 'plot', 'verify'])
    parser.add_argument("--plot_type", help="shows a plot according to the requested argument",
                        choices=['scatter', 'data-with-line', 'cost-surface-contour', 'cost-vs-iterations'])

    args = parser.parse_args()

    # retrieve arguments
    regression_type = args.type
    action = args.action
    plot_type = args.plot_type

    if regression_type == 'univariate':
        data_path = 'data/ex1data1.txt'
        col_names = ['population', 'profit']
    else:
        data_path = 'data/ex1data2.txt'
        col_names = ['size', 'bedrooms', 'price']

    # load data
    df = pd.read_csv(data_path, names=col_names, sep=',', dtype=np.float64, header=None)
    cols = df.columns
    X = df[cols[:-1]]  # define the features as the set of all columns of the dataframe except the last one
    y = df[cols[-1]]  # the last column of the dataframe corresponds to the target values

    # if regression_type is multivariate we should perform feature scaling
    if regression_type == 'multivariate':
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    else:
        scaler = None
        X = X.to_numpy()

    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1) #insert a column of 1s to the left of X. This is to account for the intercept term.

    # convert X and y to tensorflow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y.to_numpy(), dtype=tf.float32)
    y = tf.reshape(y, [y.shape[0],
                       1])  # I had to specifically force this reshape, otherwise the dimensions were misinterpreted

    # initialize model
    model = Model(X.shape[1])

    if action == 'plot':

        if not plot_type:  # deal with the case where the user has chosen to plot but hasn't specified a plot_type.
            sys.exit('Please specify a plot type. Check python lin_reg.py -h  for help.')
        else:  # the user chose to plot something. Perform processing according to plot_type and regression_type

            if regression_type == 'univariate':
                if plot_type not in ['scatter', 'data-with-line',
                                     'cost-surface']:  # these are the plots corresponding to steps 2.1 2.2 and 2.4 of the univariate regression section
                    sys.exit('The selected plot_type is not supported for this regression type.')
                else:
                    if plot_type == 'scatter':

                        plot_scatter(df)

                    elif plot_type == 'data-with-line':

                        plot_data_with_line(df)

                    elif plot_type == 'cost-surface':
                        print('plotting ', plot_type)

            else:  # regression type is multivariate
                if plot_type != 'cost-vs-iterations':  # this is the plot corresponding to step 3.2.1 of the exercise, dealing with multiple variables.
                    sys.exit('The selected plot_type is not supported for this regression type.')
                else:
                    print('plotting ', plot_type)

    else:  # action is either train/predict/verify
        if plot_type:  # deal with the case when the action is not plot but the user specified a plot_type.
            sys.exit('--plot_type cannot be used with an action other than "plot".')
        else:
            # perform the corresponding action
            if action == 'train':

                # train model
                model.train(400, X, y, learning_rate=0.01)

                # save weights
                model.save(regression_type)

            elif action == 'predict':
                if regression_type == 'univariate':
                    # make predictions for population sizes 35,000 and 70,000.
                    # initialize model from weight file
                    model.load_from_file(
                        'model/univariate.pkl')  # this will initialize the model weights self.theta to the converged weights

                    ex_x1 = tf.constant([1.0, 3.5], shape=(1, 2))
                    ex_y1 = model.predict(ex_x1).numpy()[0][0]
                    print('When the size of the population is 35,000 => The predicted profit is {:.2f}$'.format(
                        ex_y1 * 10000))

                    ex_x2 = tf.constant([1.0, 7.0], shape=(1, 2))
                    ex_y2 = model.predict(ex_x2).numpy()[0][0]
                    print('When the size of the population is 70,000 => The predicted profit is {:.2f}$'.format(
                        ex_y2 * 10000))

                else:
                    # when the regression is with multiple variables we need to take into account the feature scaling
                    # initialize model from weight file
                    model.load_from_file('model/multivariate.pkl')
                    x_multi_scaled = scaler.transform(np.array([1650, 3]).reshape(1, -1)) #use the scaler to apply the same transformation as the training data
                    ex_x_multi = tf.constant(np.concatenate([np.ones((1, 1)), x_multi_scaled], axis=1), shape=(1, 3), dtype=tf.float32) #add the 1s for the intercept and create a tensorflow constant
                    ex_y_multi = model.predict(ex_x_multi).numpy()[0][0]
                    print(
                        'When the house has an area of 1650 square feet and 3 bedrooms => Its predicted price is {:.2f}$'.format(
                            ex_y_multi))

            elif action == 'verify':
                print()
            else:
                sys.exit('Invalid action. Exiting...')
