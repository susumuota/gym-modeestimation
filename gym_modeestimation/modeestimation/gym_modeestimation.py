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

class ModeEstimationEnv(gym.Env):
    '''see https://github.com/openai/gym/blob/master/gym/core.py and https://arxiv.org/pdf/1809.09147.pdf'''

    metadata = {'render.modes': ['human']}

    N_MAX = 10
    T_MAX = 30
    NOOP = N_MAX
    R1 = T_MAX
    R2 = -T_MAX
    R3 = -T_MAX

    def __init__(self, eps=0.0):
        # gym variables
        self.observation_space = gym.spaces.Discrete(self.N_MAX) # 0-9
        self.action_space = gym.spaces.Discrete(self.N_MAX + 1) # 0-9 + NOOP
        self.reward_range = (-self.T_MAX, self.T_MAX)
        # task variables
        self.eps = eps
        self.n0 = np.random.randint(self.N_MAX)
        self.prob = self._make_prob(self.n0, self.eps, self.N_MAX)
        self.steps = 0

    def step(self, action):
        '''see equation (1) in https://arxiv.org/pdf/1809.09147.pdf'''
        self.steps += 1
        if self.steps < self.T_MAX:
            if action == self.NOOP:
                obs = np.random.choice(self.N_MAX, p=self.prob)
                return obs, 0, False, 'T <= T_MAX and No guess'
            elif action == self.n0:
                return 0, self.R1 - (self.steps - 1), True, 'T <= T_MAX and Correct guess'
            else:
                return 0, self.R2, True, 'T <= T_MAX and Incorrect guess'
        else:
            return 0, self.R3, True, 'T > T_MAX'

    def reset(self):
        self.n0 = np.random.randint(self.N_MAX)
        self.prob = self._make_prob(self.n0, self.eps, self.N_MAX)
        self.steps = 0
        obs = np.random.choice(self.N_MAX, p=self.prob)
        return obs

    def render(self, mode='human', close=False):
        print(self.eps, self.n0, self.steps, self.prob) # TODO
        return None

    def seed(self, seed=None):
        '''TODO: seeding'''
        np.random.seed(seed)
        return [seed]

    def _make_prob(self, n0, eps, n_max):
        '''see equation (5) in https://arxiv.org/pdf/1809.09147.pdf'''
        return np.array([ 1 - eps if n == n0 else eps / (n_max - 1) for n in range(n_max) ])
