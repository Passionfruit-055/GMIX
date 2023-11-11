import logging

from model.mlp import MLP

from marl.ReplayBuffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


class CommAgent(object):
    def __init__(self, config):
        self.config = config

        self.name = 'C-network'

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
        self.train_step = 0

        self.hidden_state = None
        self.target_hidden_state = None

        self.alpha = config.get("alpha", 0.001)
        self.gamma = config.get("gamma", 0.9)
        self.epsilon = config.get("epsilon", 0.2)
        self.sigma = config.get("sigma", 3)

        buffer_size = self.max_episode_len = config.get("timestep", 1000)  # 记录的是一个episode的数据，要够长

        self.comm_round = 0

        self.AoI_history = None
        self._AoI_template = self.__generate_AoI_template()

        self.receive = np.ones((self.n_agent,))

        self.overhead = np.zeros((self.n_agent,))

        self.modes = np.zeros((self.n_agent,))
        self.rewards = np.zeros((self.n_agent,))

        self.memory = ReplayBuffer(buffer_size)

        self.states = None
        # expanded obs, pass to GMIX
        self.obs = np.zeros((self.n_agent, self.config["obs_space"]))

        self.config.update({"obs_space": self.config["obs_space"] * self.n_agent + 1})  # e_obs + mode

    def _build_model(self):
        share_param = self.config.get("share_param", False)
        share = self.config.get("share", False)
        self.action_space = action_space = 3 if not share_param else 4
        self.state_space = state_space = self.config['obs_space'] * 2 + 4 + self.n_agent if share else 0

        hidden_l1_dim = self.config["hidden_l1_dim"]
        hidden_l2_dim = self.config["hidden_l2_dim"]

        self.params = []

        self.model = MLP(state_space, action_space, hidden_l1_dim, hidden_l2_dim).to(self.device)
        self.target_model = MLP(state_space, action_space, hidden_l1_dim, hidden_l2_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.params.extend(list(self.model.parameters()))

    def communication_round(self, obs, pos, params=None, done=False):
        # add here to provide a protection
        if done:
            logger = logging.getLogger()
            logger.error("last timestep, should not be stored!")
            return

        pos = self._extract_agent_pos(obs)

        self.comm_round += 1
        states = self.state_formulation(pos, obs, params)
        # do not save first and last
        # 在这一时刻将上一个时刻的历史存入，当前时刻状态充当下一时刻状态
        if self.states is not None:
            self.memory.add(self.states, self.modes, self.rewards, states)
        self.states = states

        self.choose_actions(states)

        self.compute_reward()

        e_obs = self.expand_obs(obs, self._prepare_message(obs, params))

        self.obs = obs

        # 不涉及时序，每一个timestep都可以做更新
        self.train()

        return np.concatenate((e_obs, self.modes.reshape((self.n_agent, 1))), axis=1)

    def state_formulation(self, pos, obs, params=None):
        obs = np.array(obs, dtype=np.float32).reshape((self.n_agent, -1))
        self.obs = np.array(self.obs, dtype=np.float32).reshape((self.n_agent, -1))
        params = np.array(params, dtype=np.float32).reshape((self.n_agent, -1)) if params is not None else None

        dist = self._measure_dist(pos)
        queue_delay = self._estimate_queue_delay()
        AoI = self._evaluate_AoI(dist, queue_delay)

        states = None

        onehot = np.eye(self.n_agent)

        for sender in range(self.n_agent):
            ob, pre_ob = obs[sender], self.obs[sender]
            param = params[sender] if params is not None else None
            d_i, q_i, Ao_i, c_i = dist[sender], queue_delay[:, sender], AoI[:, sender], self.overhead[sender]

            msg_pool = np.concatenate((ob, pre_ob), axis=0)
            state = np.concatenate((np.array([np.max(d_i), np.max(q_i), np.max(Ao_i), c_i]), msg_pool), axis=0)
            state = np.hstack((state, onehot[sender])) if self.config.get("share", False) else state

            states = np.vstack((states, state)) if states is not None else state

        self._receive_permission(dist, queue_delay)
        self._compute_overhead(dist, queue_delay)

        return states

    def _extract_agent_pos(self, obs):
        scenario = self.config.get('scenario', 'mpe_reference')
        if scenario == 'mpe_reference':
            pos = obs[:, :2]
        else:
            raise NotImplementedError('Undefined pos extraction from observation')

        return pos

    def _measure_dist(self, pos):
        dist = np.zeros((self.n_agent, self.n_agent))
        for sender in range(self.n_agent):
            for receiver in range(sender + 1, self.n_agent):
                dist[sender][receiver] = np.linalg.norm(pos[sender] - pos[receiver])
                dist[receiver][sender] = dist[sender][receiver]
        return dist

    def _estimate_queue_delay(self):
        # dim=0 (行）receiver, 每一行代表msg到达的次序，不能重复
        q_d = None
        for receiver in range(self.n_agent):
            arrive_seq = np.arange(self.n_agent)
            np.random.shuffle(arrive_seq)
            i = np.where(arrive_seq == 0)[0]
            arrive_seq[receiver], arrive_seq[i] = arrive_seq[i], arrive_seq[receiver]
            q_d = np.vstack((q_d, arrive_seq)) if q_d is not None else arrive_seq
        return q_d

    def __generate_AoI_template(self):
        AoI_template = np.zeros((self.n_agent, self.n_agent))
        for sender in range(self.n_agent):
            for receiver in range(sender, self.n_agent):
                if sender == receiver:
                    AoI_template[sender][receiver] = sender + 0.5
                else:
                    AoI_template[sender][receiver] = max(sender, receiver)
                AoI_template[receiver][sender] = AoI_template[sender][receiver]
        return AoI_template

    def _evaluate_AoI(self, dist, queue, comm_round=0):
        # dim=0 行 是sender
        AoI_template = self._AoI_template

        current_AoI = np.zeros((self.n_agent, self.n_agent))
        dist_grade = np.arange(self.sigma, step=self.sigma / self.n_agent)
        for sender in range(self.n_agent):
            for receiver in range(self.n_agent):
                dist_index = np.where(dist_grade >= dist[sender][receiver])[0]
                if dist_index.size == 0:
                    current_AoI[sender][receiver] = AoI_template[self.n_agent - 1][self.n_agent - 1]
                else:
                    q_index = queue[sender][receiver]
                    current_AoI[sender][receiver] = AoI_template[dist_index[0]][q_index]

        self.AoI_history = current_AoI if comm_round == 0 else self.AoI_history + current_AoI
        AoI = self.AoI_history / comm_round if comm_round != 0 else current_AoI

        return AoI

    def choose_actions(self, states):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        for a in range(self.n_agent):
            with torch.no_grad():
                q_values = self.model(states[a])
            if random.random() > self.epsilon:
                self.modes[a] = torch.argmax(q_values[0], dim=0).item()
            else:
                self.modes[a] = random.randint(0, self.action_space - 1)

        self.modes = np.array(self.modes, dtype=np.int32)

    def _prepare_message(self, obs, params=None):
        actions = self.modes.tolist()
        assert params is None, "to be developed!"
        msgs = []
        for sender in range(self.n_agent):
            action, ob, pre_ob = actions[sender], obs[sender], self.obs[sender]
            blank_msg = np.zeros(ob.shape)
            param = params[sender] if params is not None else None
            msg_kinds = (blank_msg, pre_ob, ob, (ob, param))
            msg = msg_kinds[action]
            msgs.append(msg)
        msgs = np.array(msgs, dtype=np.float32).reshape((self.n_agent, -1))
        return msgs

    def _receive_permission(self, dist, queue_delay):
        # 利用排队和距离来确定消息能不能发到接受者一方
        # dim=0 行 recevier
        self.receive = np.ones((self.n_agent, self.n_agent))
        for receiver in range(self.n_agent):
            for sender in range(self.n_agent):
                # failed when too far or end of queue
                if dist[sender][receiver] > self.sigma or \
                        queue_delay[receiver][sender] == self.n_agent - 1:
                    self.receive[receiver][sender] = 0

        # 对角线清零
        self.receive[np.eye(self.n_agent, dtype=np.bool)] = 0

    def expand_obs(self, obs, msgs):
        e_obs = None
        blank_msg = np.zeros(obs[0].shape)
        for receiver in range(self.n_agent):
            e_ob = obs[receiver].copy()
            for sender in range(self.n_agent):
                if receiver != sender:
                    if self.receive[receiver][sender]:
                        e_ob = np.concatenate((e_ob, msgs[sender]), axis=0)
                    else:
                        e_ob = np.concatenate((e_ob, blank_msg), axis=0)
            e_obs = np.vstack((e_obs, e_ob)) if e_obs is not None else e_ob
        # 保证每个agent的obs维度一致
        return e_obs

    def compute_reward(self):
        # encourage to communicate, modes that transmit more infos score higher,AoI to give negative feedback
        o1 = 1
        o2 = 0.5
        o3 = 0.8
        AoI = np.mean(self.AoI_history, axis=1)
        rewards = o1 * self.rewards + o2 * self.modes - o3 * AoI
        self.rewards = rewards

    def _compute_overhead(self, dist, queue_delay):
        for sender in range(self.n_agent):
            d, q = dist[sender], queue_delay[:, sender]
            # 将排队的次序作为权重，主要还是以距离作为衡量通信的开销
            overhead = d * q / np.sum(q)
            self.overhead[sender] = np.sum(overhead)

    def _update_target_model(self):
        assert self.target_model_update_cycle != 0, "target_model_update_cycle must be set!"
        if self.train_step % self.config["target_model_update_cycle"] == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        self.train_step += 1
        batch_size = min(self.config.get("batch_size", 32), self.comm_round - 1)
        minibatch = self.memory.sample_batch(batch_size)
        for batch in minibatch:
            states, actions, rewards, n_states = batch

            # 不存在时序的概念，相当于batch_size = agent_size
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            n_states = torch.tensor(n_states, dtype=torch.float32).to(self.device)

            q_values = self.model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            n_q_values = self.target_model(n_states).max(1)[0].detach()

            expected_q_values = rewards + self.gamma * n_q_values
            loss = self.criterion(q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

    def load_model(self, scenario):
        logger = logging.getLogger()
        logger.info(f"Load model from scenario {scenario}")
        scenario = './chkpt/' + scenario
        self.model.load_state_dict(torch.load(scenario + '_' + self.name + '.pth'))
        self.target_model.load_state_dict(torch.load(scenario + '_' + self.name + '_target.pth'))

    def save_model(self):
        assert self.save_cycle != 0, "save_cycle must be set!"
        if self.train_step % self.config["save_cycle"] == 0:
            from datetime import datetime
            now = datetime.now()
            timestamp = now.strftime("%Y_%m_%d_%H_%M_")
            timepath = now.strftime("%m.%d")
            scenario = self.config.get('scenario', None)
            assert scenario is not None, "Undefined scenario!"
            torch.save(self.model.state_dict(),
                       f"./chkpt/{timepath}/{timestamp + self.config['scenario'] + '_' + self.name}.pth")
            torch.save(self.target_model.state_dict(),
                       f"./chkpt/{timepath}/{timestamp + self.config['scenario'] + '_' + self.name}_target.pth")


if __name__ == '__main__':
    n_agent = 5
    dist = np.random.randint(0, 4, (n_agent, n_agent))
    queue = np.random.randint(0, n_agent, (n_agent, n_agent))

    model = MLP(5, 3)
    params = model.parameters()

    a = torch.rand((3, 3))
    print(a.dtype)

    for param in params:
        type(param.data)

    param_dict = model.state_dict()

    actions = np.random.randint(0, 3, (3,))
    modes = np.zeros((3,))
    for a, m in zip(actions, modes):
        m = a
    print(modes)
    print(actions)
