"""
Deep Q network for CartPole
"""


import gym
from RL_brain import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.action_space.n)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


RL = DeepQNetwork(n_actions=env.action_space.n,  # 2
                  n_features=env.observation_space.shape[0],  # 4
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)  # epsilon-greedy 的epsilon会越来越大，policy的选择会越来越经验化

total_steps = 0


for i_episode in range(100):

    observation = env.reset()  # 取第一个observation
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # 从新编写reward，使得收敛更快
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8  # 位置越靠中间越好
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5  # 角度偏移越小越好
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
