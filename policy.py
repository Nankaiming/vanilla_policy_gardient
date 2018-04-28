import numpy as np
import tensorflow as tf


class Policy:
	def __init__(self, session, n_actions):
		self.session = session
		self.n_actions = n_actions

		with tf.variable_scope('Policy'):
			# input
			self.states = tf.placeholder(tf.float32, [None, 4])
			self.actions = tf.placeholder(tf.float32, [None, 1])
			self.advantages = tf.placeholder(tf.float32, [None, 1])
			
			# predict
			action_logits = tf.layers.dense(self.states, 2, activation = None)
			self.predictions = {
				'probabilities': tf.nn.softmax(action_logits),
				'actions': tf.argmax(action_logits, axis=1)
			}
			
			# train
			one_hot_actions = tf.one_hot(tf.cast(self.actions, tf.int32), self.n_actions)
			one_hot_actions = tf.reshape(one_hot_actions, shape=(-1, self.n_actions))
			probabilities = tf.reduce_sum(tf.multiply(tf.nn.softmax(action_logits), one_hot_actions), axis=1)
			self.loss = tf.reduce_sum(-tf.log(probabilities) * self.advantages)
			self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)
		
		self.summary = tf.summary.scalar('policy_loss', self.loss)

	def epsilon_greedy(self, state, epsilon):
		predictions = self.session.run(
			self.predictions,
			feed_dict = {
				self.states: [state]
			})
		action = predictions['actions'][0]
		return np.random.randint(self.n_actions) if np.random.uniform(0,1) < epsilon else action

	def train(self, states, actions, advantages):
		loss, _, summary = self.session.run(
			[self.loss, self.optimizer, self.summary],
			feed_dict = {
				self.states: states,
				self.actions: np.reshape(actions, [-1, 1]),
				self.advantages: np.reshape(advantages, [-1, 1])
			})
		return loss, summary