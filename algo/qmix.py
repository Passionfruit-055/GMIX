from model.rnn import RNN
from model.monotonic_mixer import QMixer
from model.mlp_nonnegative import MLP
from model.sum_mixer import VDNMixer as GMixer

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

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

        self.params = []
        self._build_model()

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

        if config["loss_func"] == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(config["loss_func"]))

        self.save_cycle = 0
        self.target_model_update_cycle = 0
        self.current_step = 0

        self.hidden_state = None

        self.alpha = config.get("alpha", 0.001)
        self.gamma = config.get("gamma", 0.9)
        self.phi = config.get("phi", 0.5)

    def _build_model(self):
        n_agent = self.config["agent_num"]

        obs_space = self.config["obs_space"]
        action_space = self.config["action_space"]
        state_space = self.config["state_space"]

        hidden_l1_dim = self.config["hidden_l1_dim"]
        hidden_l2_dim = self.config["hidden_l2_dim"]

        self.model = RNN(obs_space, action_space, hidden_l1_dim).to(self.device)
        self.target_model = RNN(obs_space, action_space, hidden_l1_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict()).to(self.device)
        self.params.extend(self.model.parameters())

        self.mixer = QMixer(n_agent, hidden_l1_dim, state_space).to(self.device)
        self.target_mixer = QMixer(n_agent, hidden_l1_dim, state_space).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.params.extend(self.mixer.parameters())

        if self.config.get("guide", False):
            self.guide = MLP(obs_space, action_space, hidden_l1_dim).to(self.device)
            self.params.extend(self.guide.parameters())
            self.gmixer = GMixer().to(self.device)
            self.params.extend(self.gmixer.parameters())

    def train(self):
        self.current_step += 1

    def update_target_model(self):
        assert self.target_model_update_cycle != 0, "target_model_update_cycle must be set!"
        if self.current_step % self.config["target_model_update_cycle"] == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def compute_actions(self, obs):
        assert self.hidden_state is not None, "Hidden state hasn't been reset!"
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        q_values, self.hidden_state = self.model(obs, self.hidden_state)
        if self.config.get("guide", False):
            g_values = self.guide(obs)
            q_values -= self.phi * g_values
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
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
        agent_num = self.config.get('agent_num')
        self.hidden_state = torch.zeros((agent_num, batch_size, self.config['obs_space']))
