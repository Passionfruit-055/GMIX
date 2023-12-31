import random
import numpy as np
import logging
from collections import deque

import torch
import torch.nn as nn

from model.rnn import RNN
from model.monotonic_mixer import QMixer
from model.mlp_nonnegative import MLP
from model.sum_mixer import VDNMixer as GMixer
from marl.ReplayBuffer import MAReplayBuffer


class QMIXAgent(object):
    def __init__(self, config):
        self.config = config

        self.name = self.config.get("algo", "QMIX")

        self.model = None
        self.target_model = None

        self.mixer = None
        self.target_mixer = None

        self.need_guide = self.config.get("guide", True)
        self.guide = None

        self.comm_net = None

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

        self.n_agent = self.config["agent_num"]

        self._build_model()

        self._add_optim()

        self._add_criterion()

        self.train_step = 0
        self.save_cycle = self.config.get("save_cycle", 0)
        self.target_model_update_cycle = self.config.get("target_model_update_cycle", 0)

        self.hidden_state = None
        self.target_hidden_state = None

        self.alpha = config.get("alpha", 0.001)
        self.beta = config.get("beta", 2)
        self.gamma = config.get("gamma", 0.9)
        self.phi = config.get("phi", 0.5)
        self.epsilon = config.get("epsilon", 0.2)

    def _build_model(self):
        n_agent = self.config["agent_num"]

        obs_space = self.obs_space = self.config["obs_space"]
        action_space = self.action_space = self.config["action_space"]
        state_space = self.state_space = self.config["state_space"]

        hidden_l1_dim = self.config["hidden_l1_dim"]
        hidden_l2_dim = self.config["hidden_l2_dim"]

        self.params = []

        self.model = RNN(obs_space, action_space, hidden_l1_dim).to(self.device)
        self.target_model = RNN(obs_space, action_space, hidden_l1_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.params.extend(self.model.parameters())

        self.mixer = QMixer(n_agent, hidden_l1_dim, state_space).to(self.device)
        self.target_mixer = QMixer(n_agent, hidden_l1_dim, state_space).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.params.extend(self.mixer.parameters())

        if self.need_guide:
            self.guide = MLP(obs_space + 1, action_space, hidden_l1_dim).to(self.device)
            self.params.extend(self.guide.parameters())
            self.gmixer = GMixer().to(self.device)
            self.params.extend(self.gmixer.parameters())

        self.buffer = MAReplayBuffer(n_agent, self.config.get('buffer_size', 128))

    def train(self, autograd_detect=False):
        losses = deque(maxlen=int(1e4))

        batch_size = min(self.config.get('batch_size', 32), self.buffer.len)

        obs, actions, rewards, n_obs, dones, states, n_states, mus = self.buffer.sample(batch_size, self.device)

        # (T, N, B, size)
        T = obs.shape[0]
        N = obs.shape[1]
        assert N == self.config["agent_num"] and N == self.buffer.agent_num, "agent_num not match!"
        B = obs.shape[2]
        assert batch_size == B, "batch_size not match!"

        def _global_rewards():
            return torch.sum(rewards, dim=1).to(torch.float32)  # 各智能体 reward 累加得到 global rewards

        rewards = _global_rewards()

        def _generate_mask():
            mask_raw = dones.reshape(T, B, N)
            mask = np.ones((T, B, 1), dtype=np.float32)
            for t in range(T):
                for b in range(B):
                    for n in range(N):
                        mask[t, b, 0] = mask_raw[t, b, n] or False
                    mask[t, b, 0] = 1 if not mask[t, b, 0] else 0
            return torch.from_numpy(mask).reshape(T, B, 1).to(self.device).to(torch.float32)

        mask = _generate_mask()

        self.reset_hidden_state(B, T)

        for t in range(T):
            eval_action_qs = []
            target_action_qs = []
            g_values = []

            for n in range(N):
                with torch.autograd.set_detect_anomaly(autograd_detect):
                    eval_qs, self.hidden_state[t + 1][n] = self.model(obs[t, n], self.hidden_state[t][n])
                    target_qs, self.target_hidden_state[t + 1][n] = self.target_model(obs[t, n],
                                                                                      self.target_hidden_state[t][n])

                eval_q = torch.gather(eval_qs, 1, actions[t, n]).squeeze(-1)
                target_q = torch.max(target_qs, 1)[0]

                if self.config.get("guide", False):
                    gs = self.guide(torch.cat([obs[t, n], mus[t, n]], dim=-1).to(torch.float32))
                    g = torch.gather(gs, 1, actions[t, n]).squeeze(-1)
                    g_values.append(g)

                eval_action_qs.append(eval_q)
                target_action_qs.append(target_q)

            distributed_qs = torch.vstack(eval_action_qs).view(B, N)
            distributed_target_qs = torch.vstack(target_action_qs).view(B, N)
            distributed_gs = torch.vstack(g_values).view(B, N) if len(g_values) > 0 else None

            tot_q = self.mixer(distributed_qs, states[t].to(torch.float32)).view(B, 1)
            target_tot_q = self.target_mixer(distributed_target_qs, n_states[t].to(torch.float32)).view(B, 1)

            td_target = rewards[t] + self.gamma * torch.mul(target_tot_q.detach(), mask[t].detach())

            loss = self.criterion(tot_q, td_target)

            if distributed_gs is not None:
                bar_mu = torch.mean(mus[t].to(torch.float32), dim=0).view(B, 1)
                tot_g = self.gmixer(distributed_gs, dim=1).detach()
                target_g = bar_mu + self.beta * torch.mean(distributed_gs)
                gloss = self.criterion(tot_g, target_g)
                loss += gloss

            def _hidden_state_detach():
                for t in range(T):
                    self.hidden_state[t] = self.hidden_state[t].detach()
                    self.target_hidden_state[t] = self.target_hidden_state[t].detach()

            _hidden_state_detach()

            self.optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(autograd_detect):
                loss.backward()
            losses.append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=10, norm_type=2)
            self.optimizer.step()
            self.train_step += 1

        return losses

    def update_target_model(self):
        assert self.target_model_update_cycle != 0, "target_model_update_cycle must be set!"
        if self.train_step % self.config["target_model_update_cycle"] == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def choose_actions(self, obs):
        assert self.hidden_state is not None, "Hidden state hasn't been reset!"

        if isinstance(obs, torch.Tensor):
            obs = obs.view(self.n_agent, -1).to(self.device)
        else:
            obs = torch.tensor(obs, dtype=torch.float32).view(self.n_agent, -1).to(self.device)

        pre_actions, Qvals = self._compute_Q_vals(obs)

        warning_signals = self._generate_warning_signals(obs.cpu().detach().numpy())

        if self.need_guide:
            Qvals = Qvals.tolist()
            obs = torch.cat([obs, torch.tensor(warning_signals, dtype=torch.float32).to(self.device)], dim=-1)
            Gvals = self._compute_G_vals(obs)
            Qvals = self._modify_Q_vals(Qvals, Gvals)

        actions = []
        for a in range(self.n_agent):
            if random.random() > self.epsilon:
                # argmax only takes tensor as input
                actions.append(torch.argmax(Qvals[a], dim=0).item())
            else:
                actions.append(random.choice(range(self.action_space)))

        return actions, warning_signals

    def _compute_Q_vals(self, obs):
        pre_actions, Qvals = [], []
        for a in range(self.n_agent):
            with torch.no_grad():
                Qval, self.hidden_state[0][a] = self.model(obs[a], self.hidden_state[0][a])
            if random.random() > self.epsilon:
                pre_actions.append(torch.argmax(Qval[0], dim=0).item())
            else:
                pre_actions.append(random.choice(range(self.action_space)))
            Qvals.append(Qval.cpu().detach())
        Qvals = torch.vstack(Qvals).reshape(self.n_agent, -1)
        return pre_actions, Qvals

    def _compute_G_vals(self, obs):
        Gvals = []
        for a in range(self.n_agent):
            with torch.no_grad():
                Gval = self.guide(obs[a])
            Gvals.append(Gval.cpu().detach().numpy())
        return Gvals

    def _modify_Q_vals(self, Qvals, Gvals):
        Qvals, Gvals = np.array(Qvals).reshape(self.n_agent, -1), np.array(Gvals).reshape(self.n_agent, -1)
        fixed_Q_vals = Qvals - self.phi * Gvals
        return torch.from_numpy(fixed_Q_vals)

    def _generate_warning_signals(self, obs):
        # warning signal include external knowledge, should customize for different scenarios
        scenario = self.config.get('mpe_scenario', None)
        assert scenario is not None, "Undefined mpe_scenario!"
        warning_signals = np.zeros((self.n_agent, 1), dtype=np.float32)
        if scenario == 'mpe_reference':
            # reference环境的风险来自于当前agent与其他agent的距离，与自己目标的距离
            # map是一个2x2的地图，定义agent之间的安全距离为1，定义agent与landmark的安全距离为1
            a2a_safe_dist = 0.5
            a2l_safe_dist = 1.5
            pos = obs[:, 0:2]
            dist = obs[:, 2:8].reshape(self.n_agent, -1, 2)
            for a1 in range(self.n_agent):
                # 智能体间，要保持安全距离，防止碰撞
                for a2 in range(a1 + 1, self.n_agent):
                    if np.linalg.norm(pos[a1] - pos[a2]) < a2a_safe_dist:
                        warning_signals[a1] += 1
                        warning_signals[a2] += 1
                # 智能体与地标间，不能超出安全距离，这里可以假设成BS和node
                for d in dist[a1]:
                    if np.linalg.norm(d) > a2l_safe_dist:
                        warning_signals[a1] += 1

        return warning_signals

    def compute_utility(self, rewards, warning_signals):
        # only for mpe_reference
        # 3 landmarks and 1 teammate, so the upper limit of warning_signal for one agent is 4, 即取值范围是 [0, 4]
        # 两坐标轴的取值范围都是 [-1,1], reward 取值范围是 [0, 4], 与上面相同所以不存在需要归一化，只用参数调整权重就行
        rewards = np.array(list(rewards.values()), dtype=np.float32).reshape(self.n_agent, -1)
        o1 = 1
        o2 = 0.5
        assert rewards.shape == warning_signals.shape, "rewards and warning_signals have different shape!"
        utilities = o1 * rewards - o2 * warning_signals
        return utilities

    def load_model(self, scenario):
        logger = logging.getLogger()
        logger.info(f"Load model from mpe_scenario {scenario}")
        scenario = './chkpt/' + scenario
        self.model.load_state_dict(torch.load(scenario + '_' + self.name + '_model.pth'))
        self.target_model.load_state_dict(torch.load(scenario + '_' + self.name + '_target_model.pth'))
        self.mixer.load_state_dict(torch.load(scenario + '_' + self.name + '_mixer.pth'))
        self.target_mixer.load_state_dict(torch.load(scenario + '_' + self.name + '_target_mixer.pth'))

    def save_model(self):
        assert self.save_cycle != 0, "save_cycle must be set!"
        if self.train_step % self.config["save_cycle"] == 0:
            from datetime import datetime
            now = datetime.now()
            timestamp = now.strftime("%Y_%m_%d_%H_%M_")
            timepath = now.strftime("%m.%d")
            scenario = self.config.get('mpe_scenario', None)
            assert scenario is not None, "Undefined mpe_scenario!"
            torch.save(self.model.state_dict(),
                       f"./chkpt/{timepath}/{timestamp + self.config['mpe_scenario'] + '_' + self.name}_model.pth")
            torch.save(self.target_model.state_dict(),
                       f"./chkpt/{timepath}/{timestamp + self.config['mpe_scenario'] + '_' + self.name}_target_model.pth")
            torch.save(self.mixer.state_dict(),
                       f"./chkpt/{timepath}/{timestamp + self.config['mpe_scenario'] + '_' + self.name}_mixer.pth")
            torch.save(self.target_mixer.state_dict(),
                       f"./chkpt/{timepath}/{timestamp + self.config['mpe_scenario'] + '_' + self.name}_target_mixer.pth")

    def reset_hidden_state(self, batch_size=1, seq_len=100):
        # hidden state 是网络需要的，和输入的维度关系不大
        agent_num = self.config.get('agent_num')
        hidden_layer_size = self.model.rnn_hidden_size
        self.hidden_state = []
        self.target_hidden_state = []
        for t in range(seq_len + 1):
            self.hidden_state.append(
                torch.zeros((agent_num, batch_size, hidden_layer_size), requires_grad=False).to(self.device))
            self.target_hidden_state.append(
                torch.zeros((agent_num, batch_size, hidden_layer_size), requires_grad=False).to(self.device))

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
        if self.config["loss"] == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(self.config["loss_func"]))

    def store_experience(self, obs, actions, rewards, n_obs, dones, states=None, n_states=None, mus=None):
        # 传入的已经是ndarray
        self.buffer.add(obs, actions, rewards, n_obs, dones, states, n_states, mus)
