import datetime, os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ddpg.util.device

_snapshot_dir = "snapshot"

if not os.path.exists(_snapshot_dir):
    pathlib.Path(_snapshot_dir).mkdir(parents=True, exist_ok=True)


def _init_uniform(layer, f=None):
    if f is None:
        f = 1 / np.sqrt(layer.weight.data.size()[0])
    torch.nn.init.uniform_(layer.weight.data, -f, f)
    torch.nn.init.uniform_(layer.bias.data, -f, f)


class _SaveLoader():
    def __init__(self, name):
        self.name = name
        self.checkpoint_file = os.path.join(_snapshot_dir, f"{name}_checkpoint")

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        print(f"{self.name} saved ({self.checkpoint_file})")

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            print(f"{self.name} checkpoint ({self.checkpoint_file}) does not exist.")
            return
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only=True))
        print(f"{self.name} loaded ({self.checkpoint_file})")


class ActorNetwork(nn.Module, _SaveLoader):
    def __init__(self, learning_rate, n_inputs, fc1_dims, fc2_dims, n_actions, name):
        nn.Module.__init__(self)
        _SaveLoader.__init__(self, name)

        self.seed = torch.manual_seed(int(datetime.datetime.now().timestamp()))
        self.checkpoint_file = os.path.join(_snapshot_dir, f"{name}_checkpoint")

        self.fc1 = nn.Linear(n_inputs, fc1_dims)
        _init_uniform(self.fc1)

        self.bn1 = nn.BatchNorm1d(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        _init_uniform(self.fc2)

        self.bn2 = nn.BatchNorm1d(fc2_dims)

        self.mu = nn.Linear(fc2_dims, n_actions)
        _init_uniform(self.mu, f=0.003)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(ddpg.util.device.device)

    def forward(self, state):
        x = self.fc1(state)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if input is 1D
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        x = self.mu(x)
        x = torch.tanh(x)  # section 7
        if x.dim() == 2:
            x = x.squeeze(0)  # Add batch dimension if input is 1D
        return x


class CriticNetwork(nn.Module, _SaveLoader):
    def __init__(self, learning_rate, n_inputs, fc1_dims, fc2_dims, n_actions, name):
        nn.Module.__init__(self)
        _SaveLoader.__init__(self, name)

        self.seed = torch.manual_seed(int(datetime.datetime.now().timestamp()))
        self.checkpoint_file = os.path.join(_snapshot_dir, f"{name}_checkpoint")

        self.fc1 = nn.Linear(n_inputs + n_actions, fc1_dims)
        _init_uniform(self.fc1)

        self.bn1 = nn.BatchNorm1d(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        _init_uniform(self.fc2)

        self.bn2 = nn.BatchNorm1d(fc2_dims)

        self.q = nn.Linear(fc2_dims, 1)
        _init_uniform(self.q, f=0.003)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(ddpg.util.device.device)

    def forward(self, state, action): # notice that critic takes both state and action.
        state_value = state
        state_value = self.fc1(torch.cat((state_value, action), 1))
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        #state_value = self.bn2(state_value)
        state_value = F.relu(state_value)

        q_value = self.q(state_value)
        return q_value
