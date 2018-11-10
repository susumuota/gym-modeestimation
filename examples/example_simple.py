# -*- coding: utf-8 -*-

import gym
import gym.spaces
import gym_modeestimation

def main():
    env = gym.make('ModeEstimationEPS00-v0')
    #env = gym.make('ModeEstimationEPS04-v0')
    obs = env.reset()
    for i in range(40):
        action = env.action_space.sample()
        #action = env.NOOP
        obs, reward, done, info = env.step(action)
        print(env.t, env.n0, action, obs, reward, done, info)
        #env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == '__main__':
    main()
