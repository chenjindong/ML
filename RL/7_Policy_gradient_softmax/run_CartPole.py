"""
Policy Gradient, Reinforcement Learning.

The cart pole example

传统的Policy gradient是回合更新的

"""

import gym
import time
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    output_graph=True,
)

for i_episode in range(3000):

    observation = env.reset()
    start = time.clock()
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:  # 基于回合的学习
            ep_rs_sum = sum(RL.ep_rs)  # calculate total reward of this episode

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            print("episode:", i_episode, "  reward:", int(running_reward), " time:", time.clock()-start)

            vt = RL.learn()
            if i_episode == 10:
                # plot 第10个episode的每个step的reward
                plt.plot(vt)
                plt.xlabel('steps in this episode')
                plt.ylabel('normalized state-action value (i.e., reward)')
                plt.show()
            break

        observation = observation_