"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

Q-Table:
每一行是某个状态的四个action（上下左右）
state1 u d l r
state2 u d l r
...... . . . .
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list [0,1,2,3]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        """
        根据policy (epsilon-greedy) 产生action
        :param observation: 当前的state
        :return action: 根据policy产生的action
        """
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """
        更新Q-table
        :param s: 当前状态
        :param a: 当前状态s执行的action
        :param r: 执行action a产生的reward
        :param s_: 当前状态s执行action a进入的下一个状态
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        """
        判断state是否在Q-table中，如果不在，就把它加进去
        """
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )