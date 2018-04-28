import numpy as np
import tensorflow as tf

REWARD_DISCOUNT = 0.97

class StateValue:
	def __init__(self, session, n_actions):
		self.session = session
		self.n_actions = n_actions

		with tf.variable_scope('StateValue'):
			# input
			self.states = tf.placeholder(tf.float32, [None, 4])
			self.future_rewards = tf.placeholder(tf.float32, [None, 1])
			# predict
			h1 = tf.layers.dense(self.states, 10, activation = tf.nn.relu)
			future_rewards = tf.layers.dense(h1, 1, activation = None)
			self.predictions = {
				'future_rewards': future_rewards
			}
			# train
			self.loss = tf.nn.l2_loss(self.future_rewards - future_rewards)
			self.optimizer = tf.train.AdamOptimizer(0.1).minimize(self.loss)
		
		self.summary = tf.summary.scalar('value_loss', self.loss)

	def calculate_future_rewards(self, rewards, discount):
		n_rewards = len(rewards)
		matrix = np.zeros((n_rewards, n_rewards))
		for i_row, row in enumerate(matrix):
			for i_column, column in enumerate(row):
				if i_column - i_row >= 0:
					matrix[i_row][i_column] = np.power(discount, i_column - i_row)
		return np.dot(matrix, rewards)

	def predict(self, states):
		predictions = self.session.run(
			self.predictions, 
			feed_dict = {
				self.states: states
			})
		return predictions['future_rewards']

	def train(self, states, future_rewards):
		loss, _, summary = self.session.run(
			[self.loss, self.optimizer, self.summary], 
			feed_dict = {
				self.states: states,
				self.future_rewards: np.reshape(future_rewards, [-1, 1])
			})
		return loss, summary

