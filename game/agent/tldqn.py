from .agent import Agent
import random
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os.path
from collections import deque
import random, string

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

FILE_DQN_LEARNING_ON = 'dqn_learning_on.npz'
FIlE_DQN_LEARNING_OFF = 'dqn_learning_off.npz'


class TLDQN(object):
    def __init__(self, rows, cols, learning_rate, learning_on):
        self._sess = tf.InteractiveSession()
        foldName = randomword(5)+"_"
        # features and labels
        x = tf.placeholder(tf.float32, [None, rows, cols])
        y = tf.placeholder(tf.float32, [None, rows, cols])
        x_reshape = tf.reshape(x, [-1, rows, cols, 1])

        # TODO these are not generic but it works for reversi board size

        # transform to features vector
        network = tl.layers.InputLayer(x_reshape, name=foldName + '_input_layer')
        network = tl.layers.Conv2d(network, n_filter=32, filter_size=(4, 4), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name=foldName + 'cnn1')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                      padding='SAME', name=foldName + 'pool1')

        network = tl.layers.Conv2d(network, n_filter=64, filter_size=(4, 4), strides=(1, 1),
                                   act=tf.nn.relu, padding='SAME', name=foldName + 'cnn2')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                      padding='SAME', name=foldName + 'pool2')

        network = tl.layers.FlattenLayer(network, name=foldName + 'flatten')
        #network = tl.layers.DropoutLayer(network, keep=0.9, name=foldName + 'drop1')

        # transform to Deep Learning Layer
        network = tl.layers.DenseLayer(network, n_units=128,W_init=tf.random_uniform_initializer(0, 0.01), b_init=None,
                                       act=tf.nn.relu, name=foldName + 'relu1')
        #network = tl.layers.DropoutLayer(network, keep=0.8, name=foldName + 'drop2')
        network = tl.layers.DenseLayer(network, n_units=128,W_init=tf.random_uniform_initializer(0, 0.01), b_init=None,
                                       act=tf.nn.relu, name=foldName + 'relu2')
        #network = tl.layers.DropoutLayer(network, keep=0.8, name=foldName + 'drop3')
        network = tl.layers.DenseLayer(network, n_units=64,
                                       act=tf.identity,
                                       name=foldName +'output')
        prediction =  tf.nn.tanh(network.outputs)
        y1 = tf.reshape(y, [-1, 64])
        #cost = tl.cost.mean_squared_error(prediction, y1, is_mean=False)  # tf.reduce_sum(tf.square(nextQ - y))
        cost = tf.reduce_mean(tf.nn.l2_loss(prediction - y1))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                          epsilon=1e-08, use_locking=False).minimize(cost)
        self._network = network

        self._x = x
        self._y = y
        self._prediction = tf.reshape(prediction, [rows, cols])
        self._optimizer = optimizer
        self._cost = cost

        tl.layers.initialize_global_variables(self._sess)

        # persistence
        self._learning_on = learning_on
        self._persistence_file = FILE_DQN_LEARNING_ON if learning_on else FIlE_DQN_LEARNING_OFF
        self.load(self._persistence_file)

    def load(self, fileName):
        print('Restoring from ' + fileName)
        if os.path.isfile(fileName):
            load_params = tl.files.load_npz(path='', name=fileName)
            tl.files.assign_params(self._sess, load_params, self._network)
            print('Restored ok')

    def save(self):
        if self._learning_on:
            tl.files.save_npz(self._network.all_params, name=FILE_DQN_LEARNING_ON)

    def train(self, x, y):
        #cost = self._sess.run([self._optimizer, self._cost], feed_dict={self._x: x, self._y: y})
        _, cost = self._sess.run([self._optimizer, self._cost], feed_dict={self._x: x, self._y: y})
        return cost

    def predict(self, x):
        return self._sess.run(self._prediction, feed_dict={self._x: x})


class TLDQNAgent(Agent):
    """ Deep Q Network Agent~
    It uses the Q-learning with Deep Learning as Q-function approximation.
    """
    def __init__(self, rows, cols,
                 sign,
                 learning_on=True,
                 learning_rate=0.0001,
                 alpha=0.1,
                 gamma=1.0,
                 epsilon=0.0):
        self._dqn = TLDQN(rows, cols, learning_rate, learning_on)
        self._sign = sign
        self._learning_on = learning_on
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._costs = []
        self._x = deque([], 200)
        self._y = deque([], 200)
        self._a = deque([], 200)
        self._min_batch_size = 200
        self._prev_action = None
        self._prev_q_values = None

    def decide(self, env, state):
        valid_actions = env.valid_actions(state)
        if len(valid_actions) == 0:
            return None

        q_states = state.board.data(self._sign)
        q_values = self._dqn.predict([q_states])

        # limit the q_value between -1 and 1
        # -1 is for invalid move
        rows, cols = state.board.rows, state.board.cols
        for row in range(rows):
            for col in range(cols):
                if (row, col) in valid_actions:
                    q_values[row][col] = min(max(q_values[row][col], -0.99), 1.0)
                else:
                    q_values[row][col] = -1.0

        # greedy
        max_q_value_index = np.argmax(q_values)
        action = max_q_value_index // rows, max_q_value_index % cols

        # exploration by epsilon
        if random.random() < self._epsilon:
            action = random.choice(list(valid_actions))

        # update q_value for the last action and train network
        if self._learning_on:
            # update
            if self._prev_action is not None:
                prev_row, prev_col = self._prev_action
                self._prev_q_values[prev_row][prev_col] = \
                    (1.0 - self._alpha) * self._prev_q_values[prev_row][prev_col] + \
                    self._alpha * (self._gamma * np.max(np.max(q_values)))

            # train q-func
            if len(self._x) >= self._min_batch_size:
                self._costs.append(self._dqn.train(self._x, self._y))

            # append the new states and values for the next training
            self._x.append(q_states)
            self._y.append(q_values)
            self._a.append(action)
            self._prev_q_values = q_values
            self._prev_action = action

        return action

    def end(self, winner):
        if self._learning_on:
            # update the last move with reward
            reward = 0.0 if winner is None else 1.0 if winner == self else -1.0
            prev_row, prev_col = self._prev_action
            self._prev_q_values[prev_row][prev_col] = \
                (1.0 - self._alpha) * self._prev_q_values[prev_row][prev_col] + \
                self._alpha * reward
            self._costs.append(self._dqn.train(self._x, self._y))
            print ('Epsilon: {:.5f} Cost: {:.5f}'.format(self._epsilon, sum(self._costs)/len(self._costs)))
            self._costs = []
            self._epsilon = max(self._epsilon - 0.001, 0.0)
            self._dqn.save()
            self._prev_action = None
            self._prev_q_values = None