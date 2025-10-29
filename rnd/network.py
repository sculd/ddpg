import torch 
import torch.nn

import sac.utils


class QNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_hid):
        super(QNet, self).__init__()
        self.trunk = sac.utils.mlp(in_dim, n_hid, out_dim, 1)
        
    def forward(self, obs):
        return self.trunk(obs)


class RND(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_hid):
        super(RND, self).__init__()
        self.target_trunk = sac.utils.mlp(in_dim, n_hid, out_dim, 2)
        self.model_trunk = sac.utils.mlp(in_dim, n_hid, out_dim, 2)
        self.optimizer = torch.optim.Adam(self.model_trunk.parameters(), lr=0.0001)
        
    def get_reward(self, obs):
        y_true = self.target_trunk(obs).detach()
        y_pred = self.model_trunk(obs)

        reward = torch.pow(y_pred - y_true, 2).sum()
        return reward
    
    def update(self, r_i):
        r_i.sum().backward()
        self.optimizer.step()
        
