from model.mlp import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


class CommAgent(object):
    def __init__(self, config):
        self.config = config

        self.model = None
        self.target_model = None

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
        self.gamma = config.get("gamma", 0.9)
        self.epsilon = config.get("epsilon", 0.2)
        self.sigma = config.get("sigma", 3)

        self.max_episode = config.get("max_episode", 1000)

    def _build_model(self):

        self.action_space = action_space = self.config["action_space"]
        self.state_space = state_space = self.config["state_space"]

        hidden_l1_dim = self.config["hidden_l1_dim"]
        hidden_l2_dim = self.config["hidden_l2_dim"]

        self.params = []

        self.model = MLP(state_space, action_space, hidden_l1_dim, hidden_l2_dim).to(self.device)
        self.target_model = MLP(state_space, action_space, hidden_l1_dim, hidden_l2_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict()).to(self.device)
        self.params.extend(list(self.model.parameters()))

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

    def _state_formulation(self, pos, obs, pre_hat_obs, params=None):
        obs = np.array(obs, dtype=np.float32).view((self.n_agent, -1))
        pre_hat_obs = np.array(pre_hat_obs, dtype=np.float32).view((self.n_agent, -1))
        params = np.array(params, dtype=np.float32).view((self.n_agent, -1)) if params is not None else None

        dist = self._measure_dist(pos)
        queue_delay = self._estimate_queue_delay()
        AoI = np.zeros((self.n_agent, self.n_agent))

        for sender in range(self.n_agent):
            ob, pre_hat_ob = obs[sender], pre_hat_obs[sender]
            param = params[sender] if params is not None else None
            msg = np.concatenate((ob, pre_hat_ob, param), axis=0)

    def _measure_dist(self, pos):
        dist = np.zeros((self.n_agent, self.n_agent))
        for sender in range(self.n_agent):
            for receiver in range(sender + 1, self.n_agent):
                dist[sender][receiver] = np.linalg.norm(pos[sender] - pos[receiver])
                dist[receiver][sender] = dist[sender][receiver]
        return dist

    def _estimate_queue_delay(self):
        queue = None
        for sender in range(self.n_agent):
            arrive_seq = np.arange(self.n_agent)
            np.random.shuffle(arrive_seq)
            i = np.where(arrive_seq == 0)[0]
            arrive_seq[sender], arrive_seq[i] = arrive_seq[i], arrive_seq[sender]
            if queue is not None:
                queue = np.vstack((queue, arrive_seq))
            else:
                queue = arrive_seq
        return queue

    def _evaluate_AoI(self, msg, queue):
        AoI = np.zeros((self.n_agent, self.n_agent))
        for sender in range(self.n_agent):
            for receiver in range(self.n_agent):
                if sender == receiver:
                    continue
                else:
                    AoI[sender][receiver] = np.sum(queue[receiver] == msg[sender])
        return AoI

    def _choose_actions(self, states):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = []
        for a in range(self.n_agent):
            q_values = self.model(states[a])
            if random.random() > self.epsilon:
                actions.append(torch.argmax(q_values[0], dim=0).item())
            else:
                actions.append(random.choice(range(self.action_space)))
        return actions

    def _compute_reward(self, rewards):
        pass

    def _compute_overhead(self, rewards, ):
        pass

    def _update_target_model(self):
        assert self.target_model_update_cycle != 0, "target_model_update_cycle must be set!"
        if self.current_step % self.config["target_model_update_cycle"] == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        pass


if __name__ == '__main__':
    pass
