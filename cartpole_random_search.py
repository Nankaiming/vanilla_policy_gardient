import numpy as np
import gym
import matplotlib.pyplot as plt


def linear_action(observation, parameters):
    return 1 if np.dot(observation, parameters) > 0 else 0

def run_episode(env, pick_action, parameters, render=False):
    observation = env.reset()
    total_reward = 0
    for _ in range(200):
        if render:
            env.render()
        action = pick_action(observation, parameters)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            return total_reward

def random_search(env):
    best_reward = 0
    best_parameters = None
    for time_step in range(1000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, linear_action, parameters)
        if reward > best_reward:
            best_reward = reward
            best_parameters = parameters
        if reward == 200:
            break
    return best_parameters, time_step

env = gym.make('CartPole-v0')

time_steps = []
for i_experiment in range(100):
    parameters, time_step = random_search(env)
    time_steps.append(time_step)
    print('Experiment {}'.format(i_experiment))
print('Average time-steps {}'.format(np.average(time_steps)))
plt.hist(time_steps, 100)
plt.show()

#parameters, time_step = random_search(env)
#reward = run_episode(env, linear_action, parameters, render=True)
#print('Reward {} after time-step {}'.format(reward, time_step))

