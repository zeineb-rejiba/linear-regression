import tensorflow as tf


class Model(object):

    def __init__(self, vector_size):
        """
        initializes a linear regression model with a weight vector theta
        :param vector_size: (n+1)-dimensional weight vector, where n is the number of features
        """
        # Initialize weights to random values
        # self.theta = tf.Variable(tf.random.normal([vector_size, 1],
        #                                         mean=0.0))  # (n+1)-dimensional weight vector, where n is the number of features

        self.theta = tf.Variable(tf.zeros([vector_size, 1]))
        self.losses = []

    def set_weights(self, theta):
        """
        set weights using specified theta vector
        :param theta: a vector of weights
        """
        self.theta = theta

    def predict(self, x):
        """
        performs a prediction for the given input, using the current weight values
        :param x: an m by (n+1) matrix containing the inputs, where m is the number of rows
        :return: predicted value
        """
        y_hat = tf.matmul(x, self.theta)
        return y_hat

    def calculate_loss(self, target_y, predicted_y):
        """
        calculates the mean squared loss between the target values and the predicted values
        :param target_y: the target values
        :param predicted_y: the predicted values
        :return: the mean squared loss
        """
        loss = (tf.reduce_mean(tf.square(target_y - predicted_y))) * 0.5
        self.losses.append(loss.numpy())
        return loss

    def update(self, inputs, outputs, learning_rate):
        """
        updates the weights by performing gradient descient on the entire batch of inputs
        :param inputs: an m by (n+1) matrix containing the inputs, where m is the number of rows
        :param outputs: the target values
        :param learning_rate: the learning rate
        """
        with tf.GradientTape() as t:
            y_hat = self.predict(inputs)
            current_loss = self.calculate_loss(outputs, y_hat)

        # calculate the gradient with respect to the weight vector theta
        d_loss_d_theta = t.gradient(current_loss, self.theta)

        # update the weights theta_j = theta_j - learning_rate * d_loss_d_theta
        self.theta.assign_sub(learning_rate * d_loss_d_theta)  #  assign_sub combines tf.assign and tf.sub

    def train(self, nb_iter, X, y, learning_rate):
        """
        Trains the linear model for the specified number of iterations
        :param nb_iter: number of iterations used for training
        :param X: an m by (n+1) matrix containing the inputs, where m is the number of rows
        :param y: the target values
        :param learning_rate: the learning rate
        """
        for i in range(nb_iter):
            self.update(X, y, learning_rate)
        print('Training done!')
