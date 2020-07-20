#  Working without membrane 28/12

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import random


class GridWorld(gym.Env):
    def __init__(self):
        n_rows = 20
        n_columns = 20

        self.top_left = top_left = 0  # Top left square is the reference

        self.nS = nS = n_rows * n_columns

        self.observation_space = spaces.Discrete(nS)

        self.state = random.randint(0, n_columns * n_rows)  # Initial state

        up = 0
        right = 1
        down = 2
        left = 3
        stay = 4
        self.nA = nA = 5

        self.action_space = spaces.Discrete(nA)

        top_row = np.zeros([n_columns])
        for i in range(n_columns):
            top_row[i] = i

        square = n_columns - 1
        right_column = np.zeros([n_rows])
        for i in range(n_rows):
            right_column[i] = square
            square += n_columns

        square = top_left + (n_rows - 1) * n_columns
        bottom_row = np.zeros([n_columns])
        for i in range(n_columns):
            bottom_row[i] = square
            square += 1

        square = top_left
        left_column = np.zeros(n_rows)
        for i in range(n_rows):
            left_column[i] = square
            square += n_columns

        #  one_way =

        validation_in_state = []
        for s in range(nS):
            validation_in_state.append([up,right, down, left, stay])
        self.validation_in_state = validation_in_state

        self.goal = goal = 25

        default_reward = -1
        goal_reward = 100
        membrane_reward = 1
        self.rewlist = rewlist = [default_reward, goal_reward, membrane_reward]
        self.nR = nR = len(self.rewlist)

        P = np.zeros(((nA, nS, nS)))
        reward_matrix = np.zeros((nS, nS))

        s_dimth, a_dimth, srnext_dimth = 0, 1, 2

        for s in range(nS):
            for a in range(nA):
                if a in validation_in_state[s]:

                    if a == up:
                        if s in top_row:
                            snext = s
                        else:
                            snext = s - n_rows

                    elif a == right:
                        if s in right_column:
                            snext = s
                        else:
                            snext = s + 1

                    elif a == down:
                        if s in bottom_row:
                            snext = s
                        else:
                            snext = s + n_rows

                    elif a == left:
                        if s in left_column:
                            snext = s
                        else:
                            snext = s - 1

                    elif a == stay:
                        snext = s

                    if snext == goal:
                        rnext = goal_reward
                    else:
                        rnext = default_reward

                    reward_matrix[s, snext] = rnext
                    P[a, s, snext] = 1.  # deterministic transition dynamics

                self.nextstate_nextreward_probdistrib = self.P = P
                self.reward_matrix = reward_matrix

                self.initialstate_probdistrib = np.ones(nS) * 1 / (n_rows * n_columns)

                self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = random.randint(0, self.nS - 1)
        return self.state

    def step(self, action):
        if action not in self.validation_in_state[self.state]:
            raise RuntimeError('Invalid action={} in state={}'.format(action, self.state))
        snext = np.argmax(self.P[action, self.state, :])
        reward = self.reward_matrix[self.state, snext]

        if snext == self.goal:
            snext = self.reset()

        self.state = snext
        done = False;
        info = {}
        return self.state, reward, done, info

