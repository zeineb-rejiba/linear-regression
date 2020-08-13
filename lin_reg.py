import matplotlib.pyplot as plt
import argparse, sys
import pandas as pd
from sklearn import preprocessing
from model import Model
import tensorflow as tf
import numpy as np


def plot_scatter(df):
    ax = df.plot.scatter(x='population',
                         y='profit',
                         c='Red',
                         marker='x')
    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel("Profit in $10,000s")
    plt.savefig('figs/scatter_plot_training_data')


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
        col_names = ['', '']

    # load data
    df = pd.read_csv(data_path, names=col_names, sep=',', dtype=np.float64, header=None)
    cols = df.columns
    X = df[cols[:-1]]  # define the features as the set of all columns of the dataframe except the last one
    y = df[cols[-1]]  # the last column of the dataframe corresponds to the target values

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
                        print('plotting ', plot_type)

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

                X.insert(0, 'intercept',
                         1.0)  # insert a column of 1s to the left of X. This is to account for the intercept term.
                print('x shape after insert ', X.shape)
                # if regression_type is multivariate we should perform feature scaling
                if regression_type == 'multivariate':
                    scaler = preprocessing.StandardScaler().fit(X)
                    X = scaler.transform(X)
                else:
                    X = X.to_numpy()

                # # convert X and y to tensorflow tensors
                X = tf.convert_to_tensor(X, dtype=tf.float32)
                y = tf.convert_to_tensor(y.to_numpy(), dtype=tf.float32)
                # initialize model
                model = Model(X.shape[1])
                # model.calculate_loss(y, predicted_y=model.predict(X))

                # train model
                model.train(1500, X, y, learning_rate=0.01)

                # save weights
                model.save(regression_type)
