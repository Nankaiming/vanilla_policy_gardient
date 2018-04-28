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

def find_best_parameters(env):
    best_reward = 0
    parameters = np.random.rand(4) * 2 - 1
    scale = 0.01
    for time_step in range(1000):
        new_parameters = parameters + (np.random.rand(4) * 2 - 1) * (1000 - time_step) / 1000
        reward = run_episode(env, linear_action, new_parameters)
        if reward > best_reward:
            best_reward = reward
            parameters = new_parameters
        if reward == 200:
            break
    return parameters, time_step

env = gym.make('CartPole-v0')
parameters, time_step = find_best_parameters(env)

time_steps = []
for i_experiment in range(100):
    parameters, time_step = find_best_parameters(env)
    time_steps.append(time_step)
    print('Experiment {}'.format(i_experiment))
print('Average time-steps {}'.format(np.average(time_steps)))
plt.hist(time_steps, 100)
plt.show()


#reward = run_episode(env, linear_action, parameters, render=True)
#print('Reward {} after time-step {}'.format(reward, time_step))

