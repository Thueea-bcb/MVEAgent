# -*- encoding: utf-8 -*-
'''
@File    :   MVEAgent.py
@Time    :   2023/05/31 22:13:38
@Author  :   Thueea_bcb 
@Version :   1.0
@Contact :   thueea_bcb@outlook.com
@License :   (C)Copyright 2017-2022, Tsinghua Univ
@Desc    :   None
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import normal
import copy
import numpy as np
import random
from collections import namedtuple
from tqdm import tqdm
import gym

num_episodes = 1000
gamma = 0.99
sample_collection = 10
buffer_size = int(1e5)
minibatch_size = 64
training_epoch = 100
sampled_transitions = 32
imagination_steps = 1
tau = 0.005

# Buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', \
     'episode_step', 'action_log_prob'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self): return len(self.memory)

    def add(self, *args):
        """Save a transition."""

        # Extend memory if capacity not yet reached.
        if self.__len__() < self.capacity:
            self.memory.append(None)

        # Overwrite current entry at this position.
        self.memory[self.position] = Transition(*args)

        # Increment position, cycling back to the beginning if needed.
        self.position = (self.position + 1) % self.capacity

    def random_sample(self, batch_size):
        """Retrieve a random sample of transitions."""
        assert batch_size <= self.__len__()

        return random.sample(self.memory, batch_size)

    def ordered_sample(self, batch_size):
        assert batch_size <= self.__len__()

        return self.memory[:batch_size]

    def empty(self):
        self.memory = []
        self.position = 0

    def populate_randomly(self, fraction, step=0):
        while self.__len__() < int(fraction * self.capacity):
            state = env.env.reset()
            terminal = False
            while terminal is False:
                if str(env.env.action_space)[:8] == 'Discrete':
                    action = torch.randint(env.action_size, size=(1,)).numpy()
                    next_state, reward, terminal, _ = env.env.step(action)
                elif str(env.env.action_space)[:3] == 'Box':
                    action = torch.rand(env.action_size)
                    action_scaled = env.action_low + (env.action_high - env.action_low) * action
                    next_state, reward, terminal, _ = env.env.step(action_scaled)
                    action = (action - 0.5) * 2  # action in [-1, 1]
                else:
                    print('Action space not implemented')

                step += 1
                self.add(state, action, reward, next_state, terminal, None, None)
                state = next_state
        return step


class ProcessMinibatch:
    def __init__(self, minibatch):
        self.states, self.actions, self.rewards, self.next_states, self.terminals, self.steps, self.action_log_prob \
            = [], [], [], [], [], [], []
        for transition in minibatch:
            self.states.append(transition.state)
            self.actions.append(transition.action)
            self.rewards.append(transition.reward)
            self.next_states.append(transition.next_state)
            self.terminals.append(transition.terminal)
            self.steps.append(transition.episode_step)
            self.action_log_prob.append(transition.action_log_prob)

        self.states = torch.Tensor(self.states)
        if type(self.actions[0]) == int or self.actions[0].shape[0] == 1:
            self.actions = torch.tensor(self.actions).reshape(-1, 1)
        else:
            self.actions = torch.tensor(self.actions)
        self.rewards = torch.Tensor(self.rewards).reshape(-1, 1)
        self.next_states = torch.Tensor(self.next_states)
        self.terminals = torch.Tensor(self.terminals).reshape(-1, 1)
        if self.steps[0] is not None:
            self.steps = torch.Tensor(self.steps).reshape(-1, 1)
        if self.action_log_prob[0] is not None:
            self.action_log_prob = torch.stack(self.action_log_prob)

    def standardise(self, obs_max):
        self.states /= obs_max
        self.next_states /= obs_max

# Models
class SequentialNetwork(nn.Module):
    def __init__(self, layers):
        super(SequentialNetwork, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class QnetContinuousActions(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64 + action_size, 128)
        self.fc3 = nn.Linear(128, action_size)

    # 采用了类似于Dueling的网络结构
    def forward(self, state, action):
        x1 = F.relu(self.fc1(state))
        x = torch.cat((x1, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Functions
class CommonFunctions:
    def __init__(self, net, optimiser, target_net, tau):
        self.net = net
        self.optimiser = optimiser
        if target_net is True:
            self.target_net = copy.deepcopy(net)
        self.tau = tau

    def optimise(self, loss, grad_clamp=False):
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True)

        if grad_clamp is True:
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimiser.step()

    def hard_target_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(param.data)

    def soft_target_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)


class PolicyFunction(CommonFunctions):
    def __init__(self, net, optimiser, target_net=False, tau=None):
        super().__init__(net, optimiser, target_net, tau)

    def softmax_action(self, state):
        probs = self.net(torch.from_numpy(state).float().reshape(1, -1))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action, probs.squeeze()[action].log()

    def get_policy(self, state):
        return self.net(torch.from_numpy(state).float())

    def log_prob(self, state, action):
        policy = self.net(torch.from_numpy(state).float())
        return torch.log(policy[action])        

class ValueFunction(CommonFunctions):
    def __init__(self, net, optimiser, target_net=False, tau=None, epsilon=None):
        super().__init__(net, optimiser, target_net, tau)
        self.epsilon = epsilon

    def epsilon_greedy_action(self, state, episode):

        epsilon = self.epsilon['eps_end'] + (self.epsilon['eps_start'] - self.epsilon['eps_end']) \
                  * np.exp(-episode / self.epsilon['eps_decay'])

        if np.random.rand() < epsilon:
            return random.randrange(self.net.layers[-1].out_features)
        else:
            with torch.no_grad():
                return torch.argmax(self.net(torch.from_numpy(state).float())).item()

# model-based modules
class DynamicsModel:
    def __init__(self, model, buffer, loss_func, opt, model_type='diff', reward=None, rew_opt=None):
        self.model = model
        self.buffer = buffer
        self.loss_func = loss_func
        self.opt = opt
        self.type = model_type
        self.reward = reward
        self.rew_opt = rew_opt 

    def train_model(self, epochs, minibatch_size, grad_steps=1, standardise=False, noise_std=None):

        for i in range(epochs):
            minibatch = self.buffer.random_sample(minibatch_size)
            t = ProcessMinibatch(minibatch)

            if self.type == 'forward':
                target = t.next_states
            else:
                target = t.next_states - t.states

            if noise_std is not None:
                target += torch.normal(0, noise_std, size=t.states.shape)
                t.states += torch.normal(0, noise_std, size=t.states.shape)
                t.actions += torch.normal(0, noise_std, size=t.actions.shape)

            for _ in range(grad_steps):
                current = self.model(torch.cat((t.states, t.actions), dim=1))
                loss = self.loss_func(current, target)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def train_reward_fnc(self, epochs, minibatch_size):

        for i in range(epochs):
            minibatch = self.buffer.random_sample(minibatch_size)
            t = ProcessMinibatch(minibatch)
            target = t.rewards
            current = self.reward(torch.cat((t.states, t.actions), dim=1))
            loss = self.loss_func(current, target)
            self.rew_opt.zero_grad()
            loss.backward()
            self.rew_opt.step()


class MVEAgent(object):
    def __init__(self, obs_size, action_size):
        # pre-define
        self.obs_size, self.action_size = obs_size, action_size
        policy_layers = [nn.Linear(obs_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), \
            nn.Linear(128, action_size), nn.Tanh()]
        model_layers =  [nn.Linear(obs_size + action_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), \
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, obs_size)]
        rew_layers = [nn.Linear(obs_size + action_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), \
            nn.Linear(64, 1), nn.Tanh()]
        model_lr, policy_lr, value_lr = 1e-3, 5e-4, 1e-3

        # buffer
        self.buffer = ReplayMemory(buffer_size)

        # networks
        self.policy_net = SequentialNetwork(policy_layers)
        self.value_net = QnetContinuousActions(obs_size, action_size)
        self.model_net = SequentialNetwork(model_layers)
        self.reward_net = SequentialNetwork(rew_layers)

        # action-noise
        self.action_noise = normal.Normal(0, 0.01)

        # loss and optimise
        self.model_loss_fnc = torch.nn.MSELoss()
        self.critic_loss_fnc = torch.nn.MSELoss()
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr = policy_lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr = value_lr, weight_decay = 1e-2)
        self.model_opt = optim.Adam(self.model_net.parameters(), lr = model_lr)
        self.rew_opt = optim.Adam(self.model_net.parameters(), lr = model_lr)

        # env model
        self.dynamics = DynamicsModel(self.model_net, self.buffer, self.model_loss_fnc, \
            self.model_opt, model_type='diff', reward = self.reward_net, rew_opt = self.rew_opt)
        self.actor = PolicyFunction(self.policy_net, self.policy_opt, target_net=True, tau=tau)
        self.critic = ValueFunction(self.value_net, self.value_opt, target_net=True, tau=tau)

        # clear buffer
        self.buffer.empty()
    
    def act_with_noise(self, state):
        with torch.no_grad():
            action = torch.clamp(self.actor.get_policy(state) + self.action_noise.sample([self.action_size]), -1, 1)
            # action_scaled = (env.action_low + (env.action_high - env.action_low) * (action + 1) / 2).numpy()
        return action

    def act_without_noise(self, state):
        with torch.no_grad():
            action = torch.clamp(self.actor.get_policy(state), -1, 1)
            # action_scaled = (env.action_low + (env.action_high - env.action_low) * (action + 1) / 2).numpy()
        return action

    def train_dynamic_model(self, training_epoch = 1, minibatch_size = 64):
        if len(self.buffer) >= minibatch_size:
            self.dynamics.train_model(training_epoch, minibatch_size, noise_std=0.001)
            self.dynamics.train_reward_fnc(training_epoch, minibatch_size)

    def train_agentnets(self, sampled_transitions = 32):
        # 在这一步需要引入env作为对环境的提取，同时进行一个预测操作
        # 考虑到学习到了状态转移，因此并不考虑基于环境的直接的学习
        gamma = 0.99
        for _ in range(sampled_transitions):
            minibatch = self.buffer.random_sample(1)
            t = ProcessMinibatch(minibatch)

            # train actor network
            actor_loss = self.critic.net(t.states, self.actor.net(t.states))
            self.actor.optimise(-actor_loss)
            imagine_state = t.next_states

            # train critic network
            imagination_steps = 1 # 基于模型估计的步数
            with torch.no_grad():
                for j in range(imagination_steps):
                    imagine_action = self.actor.target_net(imagine_state)
                    imagine_action_scaled = imagine_action
                    # 需要基于模型往后加入一个操作
                    # imagine_action_scaled = env.action_low + (env.action_high - env.action_low) * (imagine_action + 1) / 2
                    reward = self.dynamics.reward(torch.cat((imagine_state.flatten(),
                                                 imagine_action_scaled.flatten()),dim=-1))
                    # 这个是因为输入用了diff模式
                    imagine_next_state = imagine_state + self.dynamics.model(torch.cat((imagine_state, imagine_action),
                                                                                dim=1))

                    t.states = torch.cat((t.states, imagine_state))
                    t.actions = torch.cat((t.actions, imagine_action))
                    t.rewards = torch.cat((t.rewards, torch.Tensor([gamma ** (j + 1) * reward]).reshape(1, -1)))
                    imagine_state = imagine_next_state

                imagine_action = self.actor.target_net(imagine_state).reshape(1, -1)
                bootstrap_Q = gamma ** (imagination_steps + 1) * self.critic.target_net(imagine_state,
                                                                                            imagine_action)

            target = torch.stack([t.rewards[i:].sum() + bootstrap_Q for i in range(len(t.rewards))]).reshape(-1, 1)
            current = self.critic.net(t.states, t.actions)
            critic_loss = self.critic_loss_fnc(target, current)
            self.critic.optimise(critic_loss)

            # soft update
            self.critic.soft_target_update()
            self.actor.soft_target_update()

        pass

    def test(self):
        pass


if __name__ == '__main__':
    # Environment details
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if str(env.action_space)[:8] == 'Discrete':
        action_size = env.action_space.n
    elif str(env.action_space)[:3] == 'Box':
        action_size = env.action_space.shape[0]
        action_high = torch.Tensor(env.action_space.high)
        action_low = torch.Tensor(env.action_space.low)

    # Agent details
    mveagent = MVEAgent(obs_dim,act_dim)

    # pre-training
    # Gather data and training
    global_step = 0
    # 随机采样
    while mveagent.buffer.__len__() < int(0.1 * mveagent.buffer.capacity):
        state = env.reset()
        terminal = False
        while terminal is False:
            if str(env.env.action_space)[:8] == 'Discrete':
                action = torch.randint(action_size, size=(1,)).numpy()
                next_state, reward, terminal, _ = env.step(action)
            elif str(env.env.action_space)[:3] == 'Box':
                action = torch.rand(action_size)
                action_scaled = action_low + (action_high - action_low) * action
                next_state, reward, terminal, _ = env.step(action_scaled)
                action = (action - 0.5) * 2  # action in [-1, 1]
            else:
                print('Action space not implemented')

            mveagent.buffer.add(state, action, reward, next_state, terminal, None, None)
            state = next_state

    mveagent.dynamics.train_model(1000, minibatch_size=64, noise_std=0.001)
    mveagent.buffer.empty()

    for episode in tqdm(range(num_episodes)):
        episode_step = 0
        episode_reward = 0
        state = env.reset()
        terminal = False
        while terminal is False:
            action = mveagent.act_with_noise(state)
            action_scaled = (action_low + (action_high - action_low) * (action + 1) / 2).numpy()
            next_state, reward, terminal, _ = env.step(action_scaled)
            episode_reward += reward
            episode_step += 1
            global_step += 1
            mveagent.buffer.add(state, action, reward, next_state, terminal, None, None)
            state = next_state

            if (episode_step % sample_collection == 0 or terminal is True) and \
                len(mveagent.buffer) >= minibatch_size:
                mveagent.train_dynamic_model(10, 64)
                mveagent.train_agentnets(32)

        print({"episode_reward": episode_reward, 'episode': episode})


    # next_state, reward, terminal, _ = env.step(action_scaled)

    pass
