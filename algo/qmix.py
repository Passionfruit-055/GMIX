import random

import numpy as np

from model.rnn import RNN
from model.monotonic_mixer import QMixer
from model.mlp_nonnegative import MLP
from model.sum_mixer import VDNMixer as GMixer
from marl.ReplayBuffer import MAReplayBuffer

from .comm import CommAgent

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMIXAgent(object):
    def __init__(self, config):
        self.config = config

        self.model = None
        self.target_model = None

        self.mixer = None
        self.target_mixer = None

        self.guide = None

        self.comm_net = None

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

        self.n_agent = self.config["agent_num"]
        self._build_model()

        self._add_optim()

        self._add_criterion()

        self.save_cycle = self.config.get("save_cycle", 0)
        self.target_model_update_cycle = self.config.get("target_model_update_cycle", 0)
        self.current_step = 0

        self.hidden_state = None
        self.target_hidden_state = None

        self.alpha = config.get("alpha", 0.001)
        self.beta = config.get("beta", 2)
        self.gamma = config.get("gamma", 0.9)
        self.phi = config.get("phi", 0.5)
        self.epsilon = config.get("epsilon", 0.2)

    def _build_model(self):
        n_agent = self.config["agent_num"]

        obs_space = self.obs_space = self.config["obs_space"] + n_agent if self.config.get("share", False) else self.config["obs_space"]
        action_space = self.action_space = self.config["action_space"]
        state_space = self.state_space = self.config["state_space"]

        hidden_l1_dim = self.config["hidden_l1_dim"]
        hidden_l2_dim = self.config["hidden_l2_dim"]

        self.params = []

        self.model = RNN(obs_space, action_space, hidden_l1_dim).to(self.device)
        self.target_model = RNN(obs_space, action_space, hidden_l1_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict()).to(self.device)
        self.params.extend(self.model.parameters())

        self.mixer = QMixer(n_agent, hidden_l1_dim, state_space).to(self.device)
        self.target_mixer = QMixer(n_agent, hidden_l1_dim, state_space).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.params.extend(self.mixer.parameters())

        if self.config.get("guide", False):
            self.guide = MLP(obs_space + 1, action_space, hidden_l1_dim).to(self.device)
            self.params.extend(self.guide.parameters())
            self.gmixer = GMixer().to(self.device)
            self.params.extend(self.gmixer.parameters())

        if self.config.get("comm", False):
            self.comm_net = CommAgent(self.config)

        self.buffer = MAReplayBuffer(n_agent, self.config.get('batch_size', 128))

    def train(self):
        self.current_step += 1
        batch_size = min(self.config.get('batch_size', 32), self.buffer.len)
        # sample batch
        obs, actions, rewards, n_obs, dones, states, n_states, mus, msgs, mask = self.buffer.sample_batch(batch_size,

                                                                                                          self.device)
        rewards = torch.sum(rewards, dim=1)  # 各智能体 reward 累加得到 global rewards

        T = obs.shape[0]
        B = obs.shape[2]
        assert batch_size == B, "batch_size not match!"
        N = n_agent = obs.shape[1]
        assert n_agent == self.config["agent_num"] and n_agent == self.buffer.agent_num, "agent_num not match!"
        self._reset_hidden_state(batch_size)

        for t in range(T):
            eval_action_qs = []
            target_action_qs = []
            g_values = []
            for n in range(N):
                eval_qs, self.hidden_state = self.model(obs[t, n], self.hidden_state)
                target_qs, self.target_hidden_state = self.target_model(obs[t, n], self.target_hidden_state)

                eval_q = torch.gather(eval_qs, 1, actions[t, n].unsqueeze(1)).squeeze(-1)
                target_q = torch.max(target_qs, 1)

                if self.config.get("guide", False):
                    gs = self.guide(torch.cat([obs[t, n], mus[t, n]], dim=-1))
                    g = torch.gather(gs, 1, actions[t, n].unsqueeze(1)).squeeze(-1)
                    g_values.append(g)

                eval_action_qs.append(eval_q)
                target_action_qs.append(target_q)

            distributed_qs = torch.stack(eval_action_qs, dim=1).view(B, N)
            distributed_target_qs = torch.stack(target_action_qs, dim=1).view(B, N)
            distributed_gs = torch.stack(g_values, dim=1).view(B, N) if len(g_values) > 0 else None

            tot_q = self.mixer(distributed_qs, states[t, 0])
            target_tot_q = self.target_mixer(distributed_target_qs, n_states[t, 0])

            td_target = rewards[t] + self.gamma * torch.mul(target_tot_q.detach(), mask[t, 0].detach())

            loss = self.criterion(tot_q, td_target)

            if distributed_gs is not None:
                bar_mu = torch.mean(mus[t], dim=0).view(B, 1)
                tot_g = self.gmixer(distributed_gs, dim=1).detach().cpu()
                target_g = bar_mu + self.beta * torch.mean(distributed_gs)
                gloss = self.criterion(tot_g, target_g)
                loss += gloss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=10, norm_type=2)
            self.optimizer.step()

    def update_target_model(self):
        assert self.target_model_update_cycle != 0, "target_model_update_cycle must be set!"
        if self.current_step % self.config["target_model_update_cycle"] == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def compute_actions(self, obs):
        assert self.hidden_state is not None, "Hidden state hasn't been reset!"
        obs = torch.tensor(obs, dtype=torch.float64).view(self.n_agent, -1).to(self.device)

        actions = []
        for a in range(self.n_agent):
            with torch.no_grad():
                Qvals, self.hidden_state[a] = self.model(obs[a], self.hidden_state[a])
            if random.random() > self.epsilon:
                actions.append(torch.argmax(Qvals[0], dim=0).item())
            else:
                actions.append(random.choice(range(self.action_space)))
        return actions

    def load_model(self, scenario):
        logger = logging.getLogger()
        logger.info(f"Load model from scenario {scenario}")
        scenario = './chkpt/' + scenario
        self.model.load_state_dict(torch.load(scenario + '_model.pth'))
        self.target_model.load_state_dict(torch.load(scenario + '_target_model.pth'))
        self.mixer.load_state_dict(torch.load(scenario + '_mixer.pth'))
        self.target_mixer.load_state_dict(torch.load(scenario + '_target_mixer.pth'))

    def save_model(self):
        assert self.save_cycle != 0, "save_cycle must be set!"
        if self.current_step % self.config["save_cycle"] == 0:
            from datetime import datetime
            now = datetime.now()
            timestamp = now.strftime("%Y_%m_%d_%H_%M_")
            timepath = now.strftime("%m.%d")
            scenario = self.config.get('scenario', None)
            assert scenario is not None, "Undefined scenario!"
            torch.save(self.model.state_dict(), f"./chkpt/{timepath}/{timestamp + self.config['scenario']}_model.pth")
            torch.save(self.target_model.state_dict(),
                       f"./chkpt/{timepath}/{timestamp + self.config['scenario']}_target_model.pth")
            torch.save(self.mixer.state_dict(), f"./chkpt/{timepath}/{timestamp + self.config['scenario']}_mixer.pth")
            torch.save(self.target_mixer.state_dict(),
                       f"./chkpt/{timepath}/{timestamp + self.config['scenario']}_target_mixer.pth")

    def _reset_hidden_state(self, batch_size=1):
        # hidden state 是网络需要的，和输入的维度关系不大
        agent_num = self.config.get('agent_num')
        hidden_layer_size = self.model.hidden_state_size
        self.hidden_state = torch.zeros((agent_num, batch_size, hidden_layer_size)).to(self.device)
        self.target_hidden_state = torch.zeros((agent_num, batch_size, hidden_layer_size)).to(self.device)

    def _add_optim(self):
        config = self.config
        if config["optimizer"] == "rmsprop":
            from torch.optim import RMSprop
            self.optimizer = RMSprop(
                params=self.params,
                lr=config["alpha"])

        elif config["optimizer"] == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(
                params=self.params,
                lr=config["alpha"])
        else:
            raise ValueError("Unknown optimizer: {}".format(config["optimizer"]))

    def _add_criterion(self):
        if self.config["loss_func"] == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(self.config["loss_func"]))

    def store_experience(self, obs, actions, rewards, n_obs, dones, states=None, n_states=None, mus=None, msgs=None):
        # 默认传进来的都是list, 在这里转变为ndarray, 并对state在N维度上做扩展（复制）, 并执行类型转换
        experience = [obs, actions, rewards, n_obs, dones, states, n_states, mus, msgs]
        experience = [
            np.array(e, dtype=np.float32) if e is not None else None
            for e in experience]
        obs, actions, rewards, n_obs, dones, states, n_states, mus, msgs = experience
        self.buffer.add(obs, actions, rewards, n_obs, dones, states, n_states, mus, msgs)




if __name__ == "__main__":
    model = RNN(21, 5, 64)
    hidden_state = torch.zeros((2, 5, 64)).to('cuda:0')
    obs = torch.rand((5, 2, 5, 21)).to('cuda:0')
    for i in range(5):
        q, hidden_state = model(obs[i], hidden_state)
        print(q.shape)
