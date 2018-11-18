# -*- coding: utf-8 -*-

import gym
import gym.spaces
import gym_modeestimation
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import torchvision
import torchvision.transforms as transforms


class Accumulator:
    '''https://arxiv.org/pdf/1809.09147.pdf'''
    # time: t
    # action: i
    # global sensitivity of the accumulator across channels: 1 / S_t
    # evidence vector: k_t_i
    # accumulator channel: nu_i = sum_t(S_t * k_t_i)
    # preference vector: rho_i = exp(nu_i) / sum_i(exp(nu_i))
    # threshold: tau
    # choose action i,  if rho_i > tau

    def __init__(self, tau=0.0, S_t=1.0, n_max=10):
        self.N_MAX = n_max
        self.NOOP = self.N_MAX
        self.nu = np.zeros(self.N_MAX)
        self.tau = tau
        self.S_t = S_t

    def get_nu(self):
        return self.nu

    def get_tau(self):
        return self.tau

    def set_tau(self, tau):
        self.tau = tau

    def reset(self):
        self.nu = np.zeros(self.N_MAX)

    def accumulate(self, k_t):
        self.nu += self.S_t * k_t

    def get_rho(self):
        exp_nu = np.exp(self.nu)
        return exp_nu / np.sum(exp_nu)

    def get_action(self):
        rho = self.get_rho()
        action = np.argmax(rho) # TODO: sampling by rho?
        return action if rho[action] > self.tau else self.NOOP


class AccumulatorNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=25, output_dim=10):
        super(AccumulatorNetwork, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class AccumulatorRL(object):
    def __init__(self, input_dim=10, hidden_dim=25, output_dim=10):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #print(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.net = AccumulatorNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08) # default parameters
        self.memory = []
        self.gamma = 0.99

    def load_model(self, file_path):
        module_dir, _ = os.path.split(os.path.realpath(__file__))
        absolute_path = os.path.join(module_dir, file_path)
        return self.net.load_state_dict(torch.load(absolute_path, map_location=self.device))

    def save_model(self, file_path):
        module_dir, _ = os.path.split(os.path.realpath(__file__))
        absolute_path = os.path.join(module_dir, file_path)
        return torch.save(self.net.state_dict(), absolute_path)

    def step(self, state):
        return self.net(state)

    def get_action(self, state):
        policy, value = self.step(state)
        action, log_prob = self.policy_to_action(policy)
        return action, log_prob, value

    def policy_to_action(self, policy):
        p = dist.Categorical(policy)
        action = p.sample()
        log_prob = p.log_prob(action)
        return action, log_prob

    def numpy_to_tensor(self, state):
        return torch.from_numpy(state).float().to(self.device)

    def append_memory(self, log_prob, value, reward):
        self.memory.append([log_prob, value, reward])

    def clear_memory(self):
        del self.memory[:]
        # self.memory = []

    def get_returns(self, rewards):
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        #returns = torch.tensor(returns)
        #eps = np.finfo(np.float32).eps.item()
        #returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

    def train_model(self):
        policy_losses = []
        value_losses = []
        mem = np.array(self.memory)
        mem[:,2] = self.get_returns(mem[:,2]) # mem[:,2] == rewards
        for log_prob, value, r in mem:
            r = torch.tensor(r).to(self.device)
            policy_losses.append(-log_prob * (r - value))
            value_losses.append(F.smooth_l1_loss(value, r))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.25)
        self.optimizer.step()
        self.clear_memory()
        return loss.item()


class EvidenceNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=20, output_dim=10):
        super(EvidenceNetwork, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, output_dim)
        self.beta_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # alpha and beta must be non-negative float
        #alpha_scores = F.relu(self.alpha_head(x)) # TODO: relu? any other options?
        #beta_scores = F.relu(self.beta_head(x))
        alpha_scores = F.softplus(self.alpha_head(x)) # TODO: relu? any other options?
        beta_scores = F.softplus(self.beta_head(x))
        state_values = self.value_head(x)
        return alpha_scores, beta_scores, state_values

class EvidenceRL(object):
    def __init__(self, input_dim=4, hidden_dim=20, output_dim=10):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #print(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.net = EvidenceNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08) # default parameters
        self.memory = []
        self.gamma = 0.99

    def load_model(self, file_path):
        module_dir, _ = os.path.split(os.path.realpath(__file__))
        absolute_path = os.path.join(module_dir, file_path)
        return self.net.load_state_dict(torch.load(absolute_path, map_location=self.device))

    def save_model(self, file_path):
        module_dir, _ = os.path.split(os.path.realpath(__file__))
        absolute_path = os.path.join(module_dir, file_path)
        return torch.save(self.net.state_dict(), absolute_path)

    def step(self, state):
        return self.net(state)

    def get_action(self, state):
        alpha, beta, value = self.step(state)
        action, log_prob = self.policy_to_action(alpha, beta)
        return action, log_prob, value

    def policy_to_action(self, alpha, beta):
        # alpha and beta must be non-negative float
        eps = 1e-8 # to avoid inf and nan
        p = dist.Beta(alpha + eps, beta + eps)
        action = p.sample()
        log_prob = p.log_prob(action)
        return action, log_prob

    def numpy_to_tensor(self, state):
        return torch.from_numpy(state).float().to(self.device)

    def append_memory(self, log_prob, value, reward):
        self.memory.append([log_prob, value, reward])

    def clear_memory(self):
        del self.memory[:]
        # self.memory = []

    def get_returns(self, rewards):
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        #returns = torch.tensor(returns)
        #eps = np.finfo(np.float32).eps.item()
        #returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

    def train_model(self):
        policy_losses = []
        value_losses = []
        mem = np.array(self.memory)
        mem[:,2] = self.get_returns(mem[:,2]) # mem[:,2] == rewards
        for log_prob, value, r in mem:
            r = torch.tensor(r).to(self.device)
            policy_losses.append(-log_prob * (r - value))
            value_losses.append(F.smooth_l1_loss(value, r))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.25)
        self.optimizer.step()
        self.clear_memory()
        return loss.item()

def number_to_onehot(n, dim=10):
    onehot = np.zeros(dim)
    onehot[n] = 1.0
    return onehot

def number_to_binary(n, dim=4):
    binary = np.zeros(dim)
    for i, value in enumerate(format(n, '0{}b'.format(dim))):
        binary[i] = value
    return binary

def env_name(eps):
    return 'ModeEstimationEPS{}-v0'.format(str(int(eps * 10)).zfill(2))

def run_mc_1(env, acc, max_episode):
    def f(obs):
        k_t = number_to_onehot(obs, env.observation_space.n)
        return k_t
    episode = 0
    result_history = []
    obs = env.reset()
    k_t = f(obs)
    acc.reset()
    acc.accumulate(k_t)
    for i in range(max_episode * env.T_MAX):
        action = acc.get_action()
        obs, reward, done, info = env.step(action)
        k_t = number_to_onehot(obs, env.observation_space.n)
        acc.accumulate(k_t)
        if done:
            result_history.append([reward, env.t])
            episode += 1
            obs = env.reset()
            k_t = number_to_onehot(obs, env.observation_space.n)
            acc.reset()
            acc.accumulate(k_t)
        if episode >= max_episode:
            break
    return np.sum(np.array(result_history), axis=0) / len(result_history)

def run_mc(seed, max_episode):
    print('\n'.join([ 'seed', str(seed) ]))
    print('\t'.join([ 'episode', 'eps', 'tau', 'reward', 't' ]))
    for eps in [ 0.0, 0.2, 0.4, 0.6, 0.8 ]:
        for tau in [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]:
            env = gym.make(env_name(eps))
            env.seed(seed)
            acc = Accumulator(tau, 1.0, env.observation_space.n)
            reward, t = run_mc_1(env, acc, max_episode)
            env.close()
            print('\t'.join([ str(item) for item in [ max_episode, eps, tau, reward, t ] ]))

def train_tau_actor_critic_1(env, acc, max_episode, tau_list):
    def f(obs):
        k_t = number_to_onehot(obs, env.observation_space.n)
        return k_t
    accrl = AccumulatorRL(env.observation_space.n, int(env.observation_space.n * 2.5), len(tau_list)) # AccumulatorRL(10, 25, 10)
    episode = 0
    result_history = []
    running_loss = 0.0
    last_result = []
    obs = env.reset()
    k_t = f(obs)
    acc.reset()
    acc.accumulate(k_t)
    for i in range(max_episode * env.T_MAX):
        # decide tau

        tau_index, log_prob, value = accrl.get_action(accrl.numpy_to_tensor(k_t))
        acc.set_tau(tau_list[tau_index.item()])
        # decide action (0-9 or NOOP)
        action = acc.get_action()
        obs, reward, done, info = env.step(action)
        k_t = f(obs)
        acc.accumulate(k_t)
        accrl.append_memory(log_prob, value, reward)
        if done:
            episode += 1
            loss = accrl.train_model()
            running_loss += loss
            result_history.append([reward, env.t])
            if episode % 500 == 0:
                print('\t'.join([ str(item) for item in [ episode, running_loss / 500, acc.get_tau(), env.EPS ] ]))
                last_result = np.sum(np.array(result_history), axis=0) / len(result_history)
                result_history = []
                running_loss = 0.0
            obs = env.reset()
            k_t = f(obs)
            acc.reset()
            acc.accumulate(k_t)
        if episode >= max_episode:
            break
    return last_result

def train_tau_actor_critic(seed, max_episode):
    result_history = []
    for eps in [ 0.0, 0.2, 0.4, 0.6, 0.8 ]:
        tau_list = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
        env = gym.make(env_name(eps))
        env.seed(seed)
        acc = Accumulator(tau_list[0], 1.0, env.observation_space.n)
        reward, t = train_tau_actor_critic_1(env, acc, max_episode, tau_list)
        env.close()
        result = [ max_episode, eps, acc.get_tau(), reward, t ]
        result_history.append(result)
        print('\t'.join([ str(item) for item in result ]))
    print('\n'.join([ 'seed', str(seed) ]))
    print('\t'.join([ 'episode', 'eps', 'tau', 'reward', 't' ]))
    for result in result_history:
        print('\t'.join([ str(item) for item in result ]))

def train_f_tau_actor_critic_1(env, acc, max_episode, tau_list):
    def f(obs, dim=4):
        # k_t = number_to_onehot(obs, env.observation_space.n)
        k_t, k_t_log_prob, k_t_value = evirl.get_action(accrl.numpy_to_tensor(number_to_binary(obs, dim)))
        return k_t, k_t_log_prob, k_t_value
    binary_dim = len(format(env.observation_space.n, 'b')) # 4
    accrl = AccumulatorRL(env.observation_space.n, int(env.observation_space.n * 2.5), len(tau_list)) # AccumulatorRL(10, 25, 5)
    evirl = EvidenceRL(binary_dim, env.observation_space.n * 2, env.observation_space.n) # AccumulatorRL(4, 20, 10)
    episode = 0
    result_history = []
    running_k_t_loss = 0.0
    running_tau_loss = 0.0
    last_result = []
    obs = env.reset()
    k_t, k_t_log_prob, k_t_value = f(obs, binary_dim)
    acc.reset()
    acc.accumulate(k_t)
    for i in range(max_episode * env.T_MAX):
        # decide tau
        k_t, k_t_log_prob, k_t_value = f(obs)
        tau_index, tau_log_prob, tau_value = accrl.get_action(k_t)
        acc.set_tau(tau_list[tau_index.item()])
        # decide action (0-9 or NOOP)
        action = acc.get_action()
        obs, reward, done, info = env.step(action)
        k_t, k_t_log_prob, k_t_value = f(obs)
        acc.accumulate(k_t)
        evirl.append_memory(k_t_log_prob, k_t_value, reward)
        accrl.append_memory(tau_log_prob, tau_value, reward)
        if done:
            episode += 1
            k_t_loss = evirl.train_model()
            tau_loss = accrl.train_model()
            running_k_t_loss += k_t_loss
            running_tau_loss += tau_loss
            result_history.append([reward, env.t])
            if episode % 500 == 0:
                print('\t'.join([ str(item) for item in [ episode, running_k_t_loss / 500, running_tau_loss / 500, acc.get_tau(), env.EPS ] ]))
                last_result = np.sum(np.array(result_history), axis=0) / len(result_history)
                result_history = []
                running_k_t_loss = 0.0
                running_tau_loss = 0.0
            obs = env.reset()
            k_t, k_t_log_prob, k_t_value = f(obs)
            acc.reset()
            acc.accumulate(k_t)
        if episode >= max_episode:
            break
    return last_result

def train_f_tau_actor_critic(seed, max_episode):
    result_history = []
    for eps in [ 0.0, 0.2, 0.4, 0.6, 0.8 ]:
        tau_list = [ 0.5, 0.6, 0.7, 0.8, 0.9 ]
        env = gym.make(env_name(eps))
        env.seed(seed)
        acc = Accumulator(tau_list[0], 1.0, env.observation_space.n)
        reward, t = train_f_tau_actor_critic_1(env, acc, max_episode, tau_list)
        env.close()
        result = [ max_episode, eps, acc.get_tau(), reward, t ]
        result_history.append(result)
        print('\t'.join([ str(item) for item in result ]))
    print('\n'.join([ 'seed', str(seed) ]))
    print('\t'.join([ 'episode', 'eps', 'tau', 'reward', 't' ]))
    for result in result_history:
        print('\t'.join([ str(item) for item in result ]))

def main():
    #run_mc(0, 10000)
    #train_tau_actor_critic(0, 50000)
    train_f_tau_actor_critic(0, 10000)

if __name__ == '__main__':
    main()
