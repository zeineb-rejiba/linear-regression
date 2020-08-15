import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

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

    # define the grid over which we will calculate the loss/cost function J
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
    """
    plots the cost for different learning rates
    :param X: the inputs for the training data (with a column of 1s added)
    :param y: the target values
    """
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