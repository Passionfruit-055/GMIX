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

    def add(self, obs, actions, rewards, n_obs, dones, states=None, n_states=None, mus=None):
        experience = (obs, actions, rewards, n_obs, dones, states, n_states, mus)  # (agent_num, T, 8)
        self.buffer.append(experience)
        self.len = len(self.buffer)

    def sample(self, batch_size, dev):
        assert batch_size <= self.len, "batch_size should be less than buffer size"

        indices = np.random.choice(self.len, batch_size, replace=False)

        minibatch = [[] for _ in range(len(self.buffer[0]))]
        for index in indices:
            episode = self.buffer[index]  # (T, N ,size)
            for d, mb in zip(episode, minibatch):
                mb.append(torch.tensor(d).to(dev) if d is not None else d)

        minibatch = [pad_sequence(mb, padding_value=0, batch_first=False) for mb in minibatch]

        minibatch = [mb.transpose(1, 2) if mb[0] is not None and len(mb[0].shape) > 2 else mb for mb in minibatch]

        return minibatch  # (T, N, B, size)

    def clear(self):
        self.buffer.clear()


class ReplayBuffer(object):
    def __init__(self, buffer_size=64):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.len = len(self.buffer)

    def add(self, states, actions, rewards, n_states):
        experience = [states, actions, rewards, n_states]
        self.buffer.append(experience)
        self.len = len(self.buffer)

    def sample_batch(self, batch_size):
        assert batch_size <= self.len, "batch_size should be less than memory size"
        indices = np.random.choice(self.len, batch_size, replace=False)
        return [self.buffer[index] for index in indices]

    def clear(self):
        self.buffer.clear()
        self.len = len(self.buffer)


