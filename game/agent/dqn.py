from .agent import Agent
import random
import numpy as np
import tensorflow as tf
import os.path
from collections import deque

FILE_DQN_LEARNING_ON = 'dqn_learning_on'
FIlE_DQN_LEARNING_OFF = 'dqn_learning_off'


class DQN(object):
    def __init__(self, rows, cols, learning_rate, learning_on):
        # features and labels
        x = tf.placeholder(tf.float32, [None, rows, cols])
        y = tf.placeholder(tf.float32, [None, rows, cols])
        # TODO these are not generic but it works for reversi board size
        # 1st convolution
        conv_s1 = 4
        conv_f1 = 32
        conv_w1 = tf.Variable(tf.truncated_normal([conv_s1, conv_s1, 1, conv_f1]))
        conv_b1 = tf.Variable(tf.truncated_normal([conv_f1]))
        x1 = tf.reshape(x, [-1, rows, cols, 1])
        c1 = tf.nn.relu(tf.nn.conv2d(x1, conv_w1, strides=[1, 1, 1, 1], padding='SAME') + conv_b1)
        # 2nd convolution
        conv_s2 = 4
        conv_f2 = 16
        conv_w2 = tf.Variable(tf.truncated_normal([conv_s2, conv_s2, conv_f1, conv_f2]))
        conv_b2 = tf.Variable(tf.truncated_normal([conv_f2]))
        c2 = tf.nn.relu(tf.nn.conv2d(c1, conv_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_b2)
        c2_size = np.product([s.value for s in c2.get_shape()[1:]])
        # 1st fully connected
        size = rows * cols
        w1 = tf.Variable(tf.truncated_normal([c2_size, size]))
        b1 = tf.Variable(tf.truncated_normal([size]))
        h1 = tf.nn.relu(tf.matmul(tf.reshape(c2, [-1, c2_size]), w1) + b1)
        # 2nd fully connected
        w2 = tf.Variable(tf.truncated_normal([size, size]))
        b2 = tf.Variable(tf.truncated_normal([size]))
        h1_drop = tf.nn.dropout(h1, 0.6)
        h2 = tf.nn.relu(tf.matmul(h1_drop, w2) + b2)
        # output layer
        w3 = tf.Variable(tf.truncated_normal([size, size]))
        b3 = tf.Variable(tf.truncated_normal([size]))
        prediction = tf.nn.tanh(tf.matmul(h2, w3) + b3)
        # optimization
        y1 = tf.reshape(y, [-1, size])
        cost = tf.reduce_mean(tf.nn.l2_loss(prediction - y1))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        # for later use
        self._x = x
        self._y = y
        self._prediction = tf.reshape(prediction, [rows, cols])
        self._optimizer = optimizer
        self._cost = cost
        # session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        self._sess.run(tf.initialize_all_variables())
        # persistence
        self._learning_on = learning_on
        self._saver = tf.train.Saver([conv_w1, conv_b1, conv_w2, conv_b2, w1, b1, w2, b2, w3, b3])
        self._persistence_file = FILE_DQN_LEARNING_ON if learning_on else FILE_DQN_LEARNING_ON
        self.load()

    def load(self):
        print('Restoring from ' + self._persistence_file)
        if os.path.isfile(self._persistence_file):
            print('Restoring Successfully')
            self._saver.restore(self._sess, self._persistence_file)

    def save(self):
        if self._learning_on:
            print('Saving to '+ self._persistence_file)
            self._saver.save(self._sess, self._persistence_file)

    def train(self, x, y):
        _, cost = self._sess.run([self._optimizer, self._cost], feed_dict={self._x: x, self._y: y})
        return cost

    def predict(self, x):
        return self._sess.run(self._prediction, feed_dict={self._x: x})


class DQNAgent(Agent):
    """ Deep Q Network Agent
    It uses the Q-learning with Deep Learning as Q-function approximation.
    """
    def __init__(self, rows, cols,
                 sign,
                 learning_on=True,
                 learning_rate=0.0001,
                 alpha=0.1,
                 gamma=1.0,
                 epsilon=0.0):
        self._dqn = DQN(rows, cols, learning_rate, learning_on)
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
            print ('Epsilon: {:.3f} Cost: {:.2f}'.format(self._epsilon, sum(self._costs)/len(self._costs)))
            self._costs = []
            self._epsilon = max(self._epsilon - 0.001, 0.0)
            self._dqn.save()
            self._prev_action = None
            self._prev_q_values = None