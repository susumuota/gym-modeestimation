# -*- coding: utf-8 -*-

# Copyright 2018 Susumu OTA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym
import gym.spaces

class ModeEstimationEnv(gym.Env):
    '''see https://github.com/openai/gym/blob/master/gym/core.py and https://arxiv.org/pdf/1809.09147.pdf'''

    metadata = {'render.modes': ['human']}

    def __init__(self, eps=0.0, observation_type=None, n_max=10, t_max=30):
        # task constants
        self.EPS = eps
        self.N_MAX = n_max
        self.T_MAX = t_max
        self.NOOP = self.N_MAX
        self.R1 = self.T_MAX
        self.R2 = -self.T_MAX
        self.R3 = -self.T_MAX
        self.BINARY_DIM = len(format(self.N_MAX, 'b')) # 4
        # task variables
        self.n0 = np.random.randint(self.N_MAX)
        self.p = self._make_prob(self.n0, self.EPS, self.N_MAX)
        self.t = 0
        self.observation_type = observation_type
        # gym variables
        self.observation_space = gym.spaces.Discrete(self._get_observation_space_n(self.observation_type))
        self.action_space = gym.spaces.Discrete(self.N_MAX + 1) # 0-9 + NOOP
        self.reward_range = (-self.T_MAX, self.T_MAX)

    def step(self, action):
        '''see equation (1) in https://arxiv.org/pdf/1809.09147.pdf'''
        self.t += 1
        n = np.random.choice(self.N_MAX, p=self.p)
        obs = self._number_to_observation(n, self.observation_type)
        if self.t < self.T_MAX:
            if action == self.NOOP:
                return obs, 0, False, 't <= T_max and No guess'
            elif action == self.n0:
                return obs, self.R1 - (self.t - 1), True, 't <= T_max and Correct guess'
            else:
                return obs, self.R2, True, 't <= T_max and Incorrect guess'
        else:
            return obs, self.R3, True, 't > T_max'

    def reset(self):
        self.n0 = np.random.randint(self.N_MAX)
        self.p = self._make_prob(self.n0, self.EPS, self.N_MAX)
        self.t = 0
        n = np.random.choice(self.N_MAX, p=self.p)
        obs = self._number_to_observation(n, self.observation_type)
        return obs

    def render(self, mode='human', close=False):
        print(self.EPS, self.n0, self.t, self.p) # TODO
        return None

    def seed(self, seed=None):
        '''TODO: seeding'''
        # np.random.seed(seed)
        return [seed]

    def _make_prob(self, n0, eps, n_max):
        '''see equation (5) in https://arxiv.org/pdf/1809.09147.pdf'''
        return np.array([ 1 - eps if n == n0 else eps / (n_max - 1) for n in range(n_max) ])

    def _get_observation_space_n(self, observation_type=None):
        if observation_type == 'onehot':
            return self.N_MAX
        elif observation_type == 'binary':
            return self.BINARY_DIM
        else:
            return 1


    def _number_to_observation(self, n, observation_type=None):
        if observation_type == 'onehot':
            return self._number_to_onehot(n, self.N_MAX)
        elif observation_type == 'binary':
            return self._number_to_binary(n, self.BINARY_DIM)
        else:
            return np.array([n])

    def _number_to_onehot(self, n, dim=10):
        onehot = np.zeros(dim)
        onehot[n] = 1.0
        return onehot

    def _number_to_binary(self, n, dim=4):
        binary = np.zeros(dim)
        for i, value in enumerate(format(n, '0{}b'.format(dim))):
            binary[i] = value
        return binary
