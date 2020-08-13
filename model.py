import tensorflow as tf
import pickle


class Model(object):

    def __init__(self, vector_size):
        # Initialize weights to random values
        self.theta = tf.Variable(tf.random.normal([vector_size, 1],
                                                  mean=0.0))  # (n+1)-dimensional weight vector, where n is the number of features

        # self.theta = tf.zeros([vector_size, 1])

    def predict(self, x):
        y_hat = tf.matmul(x, self.theta)
        return y_hat

    def calculate_loss(self, target_y, predicted_y):
        loss = (tf.reduce_mean(tf.square(target_y - predicted_y))) * 0.5
        print('Loss=%2.5f' % (loss))
        return loss

    def update(self, inputs, outputs, learning_rate):
        with tf.GradientTape() as t:
            y_hat = self.predict(inputs)
            current_loss = self.calculate_loss(outputs, y_hat)

        # calculate the gradient with respect to the weight vector theta
        d_loss_d_theta = t.gradient(current_loss, self.theta)

        # update the weights
        # self.theta.assign_sub(learning_rate * d_loss_d_theta) # combines tf.assign and tf.sub
        self.theta.assign_sub(learning_rate * d_loss_d_theta)

    def train(self, nb_iter, X, y, learning_rate):
        for i in range(nb_iter):
            print('\nIteration= ', i)

            self.update(X, y, learning_rate)
        print('Training done!')

    def save(self, reg_type):
        save_path = 'model/' + reg_type + '.pkl'
        with open(save_path, 'wb') as w_file:
            pickle.dump(self.theta, w_file)
        print('Saving model weights to: ', save_path)
