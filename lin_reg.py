import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from model import Model


def plot_scatter(df):
    """
    creates a scatter plot from the data points in the provided data frame
    :param df: training data frame
    """
    plt.figure()
    ax = df.plot.scatter(x='population',
                         y='profit',
                         c='Red',
                         marker='x')
    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel("Profit in $10,000s")
    plt.savefig('figs/scatter_plot_training_data.png')
    print('Saving plot to figs/scatter_plot_training_data.png')


def plot_data_with_line(df, weights):
    """
    plots the training data as well as the regression line
    :param df: the training data frame
    :param weights: the weights/parameters of the regression line
    """
    weights = weights.numpy()
    slope = weights[1][0]  # theta_1
    intercept = weights[0][0]  # theta_0
    print('Slope= ', slope)
    print('Intercept= ', intercept)

    # plot the training data directly from the dataframe
    plt.figure()
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
    print('Saving plot to figs/data_with_line.png')


def plot_surface_and_contour(w, X, y):
    """
    Creates a surface and a contour plot.
    The contour part is based on the following tutorial: http://www.adeveloperdiary.com/data-science/how-to-visualize-gradient-descent-using-contour-plot-in-python/
    :param w: the converged weight vector
    :param X: the input data
    :param y: the target value
    """
    theta0 = np.linspace(-10, 10, 100)
    theta1 = np.linspace(-1, 4, 100)
    mse_vals = np.zeros(shape=(theta0.size, theta1.size))

    for i, value1 in enumerate(theta0):
        for j, value2 in enumerate(theta1):
            model_int = Model(X.shape[1])
            theta = tf.constant([value1, value2], dtype=tf.float32, shape=[2, 1])
            model_int.set_weights(theta)
            y_hat = model_int.predict(X)
            mse_vals[i, j] = model_int.calculate_loss(y, y_hat).numpy()

    # We need to transpose mse_vals or else the axes will be flipped
    mse_vals = mse_vals.T

    # plot surface
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0, theta1, mse_vals, cmap='viridis')
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Surface')

    # contour plot
    ax = plt.subplot(122)
    levels = np.logspace(-2, 3, 20)
    plt.contour(theta0, theta1, mse_vals, levels)

    # plot the converged theta values at the center
    plt.plot(w[0][0], w[1][0], 'ro')

    plt.xlabel("theta0")
    plt.ylabel("theta1")
    plt.title('Contour, showing minimum')
    plt.savefig('figs/surface_and_contour.png')
    print('Saving plot to figs/surface_and_contour.png')


def plot_cost_per_lr(X, y):

    plt.figure()
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")

    lr_list = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    losses_per_lr_list = []
    nb_iterations = 50
    for i in range(len(lr_list)):
        lr = lr_list[i]
        model = Model(X.shape[1])
        print('Training model using ', nb_iterations, 'iterations and learning rate=', lr)
        model.train(nb_iterations, X, y, learning_rate=lr)
        J_history = model.losses
        losses_per_lr_list.append(J_history)
        plt.plot(range(nb_iterations), losses_per_lr_list[i], label='alpha= {:.3f}'.format(lr))
    plt.legend()
    plt.savefig('figs/different_learning_rates')


if __name__ == "__main__":

    ###############################################
    # Step 2: Linear regression with one variable
    # Step 2.1 Plotting the data
    ###############################################
    print('Step 2: Linear regression with one variable')
    print('Step 2.1 Plotting the data')
    data_path = 'data/ex1data1.txt'
    col_names = ['population', 'profit']

    # load data
    df = pd.read_csv(data_path, names=col_names, sep=',', dtype=np.float64, header=None)

    plot_scatter(df)
    print('\n')
    ###############################################
    # Step 2.2 Gradient desent
    ###############################################
    print('Step 2.2 Gradient desent')

    cols = df.columns
    X = df[cols[:-1]]  # define the features as the set of all columns of the dataframe except the last one
    y = df[cols[-1]]  # the last column of the dataframe corresponds to the target values

    X = X.to_numpy()

    X = np.concatenate([np.ones((X.shape[0], 1)), X],
                       axis=1)  # insert a column of 1s to the left of X. This is to account for the intercept term.

    # convert X and y to tensorflow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y.to_numpy(), dtype=tf.float32)
    y = tf.reshape(y, [y.shape[0],
                       1])  # I had to specifically force this reshape, otherwise the dimensions were misinterpreted

    # initialize model
    model = Model(X.shape[1])

    # train model
    nb_iterations = 1500
    lr = 0.01
    print('Training model using ', nb_iterations, 'iterations and learning rate=', lr)
    model.train(nb_iterations, X, y, learning_rate=lr)

    ex_x1 = tf.constant([1.0, 3.5], shape=(1, 2))
    ex_y1 = model.predict(ex_x1).numpy()[0][0]
    print('When the size of the population is 35,000 => The predicted profit is {:.2f}$'.format(
        ex_y1 * 10000))

    ex_x2 = tf.constant([1.0, 7.0], shape=(1, 2))
    ex_y2 = model.predict(ex_x2).numpy()[0][0]
    print('When the size of the population is 70,000 => The predicted profit is {:.2f}$'.format(
        ex_y2 * 10000))

    print('\n')
    ###############################################
    # Step 2.3: Debugging
    ###############################################
    print('Step 2.3: Debugging')
    plot_data_with_line(df, model.theta)
    print('\n')
    ################################################
    # Step 2.4: Visualizing J(theta)
    ################################################
    print('Step 2.4: Visualizing J(theta)')

    w = model.theta.numpy()
    plot_surface_and_contour(w, X, y)

    print('\n')
    ###############################################
    # Step 3: Linear regression with multiple variables
    # Step 3.1 Feature normalization
    ###############################################
    print('Step 3: Linear regression with multiple variables')
    print('Step 3.1 Feature normalization')

    data_path = 'data/ex1data2.txt'
    col_names = ['size', 'bedrooms', 'price']

    # load data
    df = pd.read_csv(data_path, names=col_names, sep=',', dtype=np.float64, header=None)
    cols = df.columns
    X_multi = df[cols[:-1]]  # define the features as the set of all columns of the dataframe except the last one
    y = df[cols[-1]]  # the last column of the dataframe corresponds to the target values

    scaler = preprocessing.StandardScaler().fit(X_multi)
    X = scaler.transform(X_multi)

    print('\n')
    ###############################################
    # Step 3.2 Gradient descent
    ###############################################
    print('Step 3.2 Gradient descent')

    X = np.concatenate([np.ones((X.shape[0], 1)), X],
                       axis=1)  # insert a column of 1s to the left of X. This is to account for the intercept term.

    # convert X and y to tensorflow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y.to_numpy(), dtype=tf.float32)
    y = tf.reshape(y, [y.shape[0],
                       1])  # I had to specifically force this reshape, otherwise the dimensions were misinterpreted

    # initialize model
    model = Model(X.shape[1])
    lr = 0.01
    nb_iterations = 500
    print('Training model using ', nb_iterations, 'iterations and learning rate=', lr)
    model.train(nb_iterations, X, y, learning_rate=lr)

    print('\n')
    ###############################################
    # Step 3.2.1 Selecting learning rates
    ###############################################
    print('Step 3.2.1 Selecting learning rates')

    plot_cost_per_lr(X,y)

    # from the figure, it looks like lr=0.3 is the best one. So, training will be performed using this value.
    model = Model(X.shape[1])
    lr = 0.3
    nb_iterations = 500
    print('Training model using ', nb_iterations, 'iterations and learning rate=', lr)
    model.train(nb_iterations, X, y, learning_rate=lr)

    # when the regression is with multiple variables we need to take into account the feature scaling
    x_multi_scaled = scaler.transform(np.array([1650, 3]).reshape(1,
                                                                  -1))  # use the scaler to apply the same transformation as the training data
    ex_x_multi = tf.constant(np.concatenate([np.ones((1, 1)), x_multi_scaled], axis=1), shape=(1, 3),
                             dtype=tf.float32)  # add the 1s for the intercept and create a tensorflow constant
    ex_y_multi = model.predict(ex_x_multi).numpy()[0][0]
    print(
        'When the house has an area of 1650 square feet and 3 bedrooms => Its predicted price is {:.2f}$'.format(
            ex_y_multi))

    print('\n')
    ###############################################
    # Step 3.3 Normal equations
    ###############################################
    print('Step 3.3 Normal equations')

    # feature scaling is not needed, so we use X_multi which are the features before scaling.
    X = X_multi.to_numpy()
    X = np.concatenate([np.ones((X.shape[0], 1)), X],
                       axis=1)  # insert a column of 1s to the left of X. This is to account for the intercept term.
    # we use the @ operator for matrix multiplication whereas the .T corresponds to the transpose operation
    theta_neq = np.linalg.inv(X.T @ X) @ X.T @ y
    pred_neq = (np.array([1, 1650, 3]).reshape(1, -1) @ theta_neq).numpy()[0][0]
    print(
        'When the house has an area of 1650 square feet and 3 bedrooms => Its predicted price is {:.2f}$'.format(
            pred_neq))
