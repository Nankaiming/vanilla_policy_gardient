import numpy as np
import tensorflow as tf
import gym
import policy, state_value


def run_episode(sess, env, epsilon_greedy, render=False, epsilon=0.5):
	state = env.reset()
	states = []
	actions = []
	rewards = []
	for _ in range(200):
		if render:
			env.render()
		states.append(state)
		action = epsilon_greedy(state, epsilon)
		actions.append(action)
		state, reward, done, info = env.step(action)
		rewards.append(reward)
		if done:
			break
	return states, actions, rewards

env = gym.make('CartPole-v0')

session = tf.InteractiveSession()
summary_writer = tf.summary.FileWriter('summary/train24', session.graph)
policy = policy.Policy(session, env.action_space.n)
state_value = state_value.StateValue(session, env.action_space.n)
session.run(tf.global_variables_initializer())

# train
for episode in range(1):
	states, actions, rewards = run_episode(session, env, policy.epsilon_greedy)
	future_rewards = state_value.calculate_future_rewards(rewards, 0.97)
	state_value_loss, state_value_loss_summary = state_value.train(states, future_rewards)
	
	predicted_future_rewards = state_value.predict(states)
	advantages = future_rewards - predicted_future_rewards
	policy_loss, policy_loss_summary = policy.train(states, actions, advantages)

	summary_writer.add_summary(state_value_loss_summary, episode)
	summary_writer.add_summary(policy_loss_summary, episode)

	if np.sum(rewards) == 200:
		print('Episode {:5d} Reward 200'.format(episode))
	else:
		print('Episode {:5d}'.format(episode))

# render
while True:
	states, actions, rewards = run_episode(session, env, policy.epsilon_greedy, render=True, epsilon=0)
	print('Reward {}'.format(np.sum(rewards)))

env.close()