import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = [[]]
        self.new_state_memory = [[]]
        self.action_memory = [[]]
        self.reward_memory = [0]
        self.terminal_memory = [0]

    def store_transition(self, states, actions, rewards, states_, done):

        #States,actions, rewards, are lists in which the ith element are all the states,actions and reward of the ith agent respectively
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = states
        self.new_state_memory[index] = states_
        self.action_memory[index] = actions
        self.reward_memory[index] = rewards
        self.terminal_memory[index] = 1 - np.array(done)
        self.mem_cntr += 1
        self.append_all()

    def append_all(self):
        self.state_memory.append([])
        self.new_state_memory.append([])
        self.action_memory.append([])
        self.reward_memory.append(0)
        self.terminal_memory.append(0)

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size) 
        batch = np.random.choice(max_mem, batch_size)

        states,actions,states_,rewards,terminal=[],[],[],[],[]

        for i in batch:
            states.append(self.state_memory[i])
            actions.append(self.action_memory[i])
            states_.append(self.new_state_memory[i])
            rewards.append(self.reward_memory[i])
            terminal.append(self.terminal_memory[i])

        return states, actions, rewards, states_, terminal