import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer1_dim=64, layer2_dim=128, chkpt_dir='./chkpt'):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, layer1_dim)
        self.fc2 = nn.Linear(layer1_dim, layer2_dim)
        self.fc3 = nn.Linear(layer2_dim, output_dim)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
