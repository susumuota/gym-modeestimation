# gym-modeestimation

Gym environment class and examples for Mode Estimation Task.


# Links

- Origianl Paper
"Better Safe than Sorry: Evidence Accumulation Allows for Safe Reinforcement Learning",
Akshat Agarwal, Abhinau Kumar V, Kyle Dunovan, Erik Peterson, Timothy Verstynen, Katia Sycara,
2018.
https://arxiv.org/abs/1809.09147

- OpenAI Gym
https://github.com/openai/gym

- Beta Distribution
https://www.slideshare.net/matsukenbook/6-lt-59735455


# Install

```
pip install gym

git clone https://github.com/susumuota/gym-modeestimation.git
cd gym-modeestimation
pip install -e .
cd ..
```


# Uninstall

```
pip uninstall gym-modeestimation
pip uninstall gym
```


# Example

## simplest example

```python
import gym
import gym_modeestimation

def main():
    env = gym.make('ModeEstimationEPS00-v0')
    obs = env.reset()
    for i in range(40):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(env.steps, env.n0, action, obs, reward, done, info)
        if done:
            obs = env.reset()
    env.close()

if __name__ == '__main__':
    main()
```

## Available Environments

```
ModeEstimationEPS00-v0
ModeEstimationEPS02-v0
ModeEstimationEPS04-v0
ModeEstimationEPS06-v0
ModeEstimationEPS08-v0
ModeEstimationEPS10-v0
```

See https://github.com/susumuota/gym-modeestimation/blob/master/gym_modeestimation/__init__.py for more details.

You can add custom environments. See these pages.

https://github.com/openai/gym/blob/master/gym/envs/README.md
https://github.com/openai/gym/blob/master/gym/envs/__init__.py


# Author

Susumu OTA  susumu dot ota at g mail dot com

