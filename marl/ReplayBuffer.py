import random
from collections import deque

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class MAReplayBuffer(object):
    def __init__(self, agent_num, buffer_size=64):
        self.agent_num = agent_num
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.len = len(self.buffer)

    def add(self, obs, actions, rewards, n_obs, dones, states=None, n_states=None, mus=None, msgs=None):
        experience = (obs, actions, rewards, n_obs, dones, states, n_states, mus, msgs)  # (agent_num, T, 8)
        self.buffer.append(experience)
        self.len = len(self.buffer)

    def sample_batch(self, batch_size, dev):
        # 最后以不同量的形式返回，形状为 (B, N, T, size) N = agent_num, size 是这个量的维度
        assert batch_size <= self.len, "batch_size should be less than buffer size"
        indices = np.random.choice(self.len, batch_size, replace=False)
        T = max(len(self.buffer[index][0][0]) for index in indices)

        minibatch = [[] for _ in range(len(self.buffer[0]))]
        mask = []
        for index in indices:
            episode = self.buffer[index]
            for d, mb in zip(episode, minibatch):
                d = torch.tensor(d).transpose(0, 1).to(dev) if d is not None else d
                mb.append(d)
            mask.append(torch.ones(minibatch[0][-1].shape[0:2]).unsqueeze(2))

        minibatch.append(mask)

        minibatch = [pad_sequence(mb, padding_value=0).transpose(1, 2) if mb[0] is not None else mb for mb in minibatch]

        return minibatch  # (T, N, B, size)

    def clear(self):
        self.buffer.clear()


class ReplayBuffer(object):
    def __init__(self, buffer_size=64):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.len = len(self.buffer)

    def add(self, states, actions, rewards, n_states, dones):
        experience = [states, actions, n_states, rewards, dones]
        self.buffer.append(experience)
        self.len = len(self.buffer)

    def sample_batch(self, batch_size):
        assert batch_size <= self.len, "batch_size should be less than memory size"
        indices = np.random.choice(self.len, batch_size, replace=False)
        return [self.buffer[index] for index in indices]

    def clear(self):
        self.buffer.clear()
        self.len = len(self.buffer)


if __name__ == '__main__':
    obs = np.zeros((2, 30, 21))
    short_obs = np.zeros((2, 10, 21))
    actions = np.zeros((2, 30, 1))
    rewards = np.zeros((2, 30, 1))
    n_obs = np.zeros((2, 30, 21))
    dones = np.zeros((2, 30, 2))
    states = np.zeros((2, 30, 21))
    n_states = np.zeros((2, 30, 21))
    warnings = np.zeros((2, 30, 1))
    mus = np.zeros((2, 30, 1))
    msgs = None

    buffer = MAReplayBuffer(2)
    for i in range(10):
        buffer.add(random.choice([obs, short_obs]), actions, rewards, n_obs, dones, states, n_states, mus, msgs)
    minibatch = buffer.sample_batch(5, 'cuda:0')
    print(minibatch)
