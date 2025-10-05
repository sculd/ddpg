import torch
import torch.nn as nn

import sac.utils


class Sys_R(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_units, fc2_units):
        super(Sys_R, self).__init__()

        self.R = sac.utils.mlp(2 * state_dim + action_dim, int((fc1_units + fc2_units) // 2), 1, 3)
        self.apply(sac.utils.weight_init)

    def forward(self, state, next_state, action):
        sa = torch.cat([state, next_state, action], 1)
        return self.R(sa)


class SysModel(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        super(SysModel, self).__init__()

        self.SR = sac.utils.mlp(state_size + action_size, int((fc1_units + fc2_units) // 2), state_size, 3)
        self.apply(sac.utils.weight_init)
        self.outputs = dict()

    def forward(self, state, action):
        """Build a system model to predict the next state at a given state."""
        xa = torch.cat([state, action], 1)
        return self.SR(xa)
