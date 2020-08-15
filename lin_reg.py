import pandas as pd
from sklearn import preprocessing

from plot_utils import *

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
