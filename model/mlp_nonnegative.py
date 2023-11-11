import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, chkpt_dir='./chkpt'):
        super(MLP, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, output_dim), nn.ReLU())

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        output = self.fc3(x)
        return output
