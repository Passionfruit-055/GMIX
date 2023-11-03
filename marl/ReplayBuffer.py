import random
from collections import deque

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class ReplayBuffer(object):
    def __init__(self, agent_num, buffer_size=64):
        self.agent_num = agent_num
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.seq_length = deque(maxlen=buffer_size)

    def add(self, obs, actions, rewards, n_obs, dones, states=None, n_states=None, msgs=None):
        experience = (obs, actions, rewards, n_obs, dones, states, n_states, msgs)
        self.buffer.append(experience)
        self.seq_length.append(len(obs))

    def sample_batch(self, batch_size, dev):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        minibatch = self.buffer[indices]
        T = max(self.seq_length[indices])

        def _to_tensor(batch, device):
            for d in batch:
                d = torch.tensor(d).to(device)
            return batch

        return _to_tensor(minibatch, dev), T

    def clear(self):
        self.buffer.clear()

