import os
import torch 
import copy
import hydra
import torch.nn.functional as F


class DQNRNDAgent:
    def __init__(self, action_range, device, gamma, qnet_cfg, rnd_cfg):
        self.action_range = action_range
        self.device = torch.device(device)
        self.qnet = hydra.utils.instantiate(qnet_cfg).to(self.device)
        self.qnet_target = copy.deepcopy(self.qnet).to(self.device)
        self.rnd = hydra.utils.instantiate(rnd_cfg).to(self.device)

        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.qnet.parameters(),lr=0.001)
        self.batch_size = 64
        self.epsilon = 0.1
        self.step_counter = 0
        self.epsi_high = 0.9
        self.epsi_low = 0.05
        self.steps = 0
        self.count = 0
        self.decay = 200
        self.eps = self.epsi_high
        self.update_target_step = 500
        self.train(mode=True)

    def reset(self):
        pass

    def act(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        action = self.qnet(state)
        action = action.detach().cpu().numpy()
        return action

    def train(self, mode=True):
        # mode False means eval mode
        self.mode = mode
        if mode:
            self.qnet.train()
            self.qnet_target.eval()
            self.rnd.train()
        else:
            self.qnet.eval()
            self.qnet_target.eval()
            self.rnd.eval()

    def update(self, replay_buffer, logger, step):
        self.optimizer.zero_grad()

        obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)

        # Convert numpy arrays to torch tensors
        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        # Ensure action is the right shape for gather
        if action.dim() == 1:
            action = action.unsqueeze(1)

        # Get intrinsic reward and update RND
        Ri = self.rnd.get_reward(obs)
        self.rnd.update(Ri)

        # Compute Q-learning loss
        target_q = reward.squeeze() + self.gamma*self.qnet_target(next_obs).max(dim=1)[0].detach()*not_done.squeeze()
        policy_q = self.qnet(obs).gather(1, action)
        loss = F.smooth_l1_loss(policy_q.squeeze(), target_q)
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def save(self, filepath, save_optimizers=False):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'qnet_state_dict': self.qnet.state_dict(),
            'qnet_target_state_dict': self.qnet_target.state_dict(),
            'rnd_target_trunk_state_dict': self.rnd.target_trunk.state_dict(),
            'rnd_model_trunk_state_dict': self.rnd.model_trunk.state_dict(),
            'step_counter': self.step_counter,
            'steps': self.steps,
            'eps': self.eps,
        }

        if save_optimizers:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'rnd_optimizer_state_dict': self.rnd.optimizer.state_dict(),
            })

        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath} (optimizers={'included' if save_optimizers else 'excluded'})")

    def load(self, filepath, load_optimizers=False):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Load model weights
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.qnet_target.load_state_dict(checkpoint['qnet_target_state_dict'])
        self.rnd.target_trunk.load_state_dict(checkpoint['rnd_target_trunk_state_dict'])
        self.rnd.model_trunk.load_state_dict(checkpoint['rnd_model_trunk_state_dict'])

        # Load training state
        self.step_counter = checkpoint.get('step_counter', 0)
        self.steps = checkpoint.get('steps', 0)
        self.eps = checkpoint.get('eps', self.epsi_high)

        # Load optimizer states if requested and available
        if load_optimizers:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'rnd_optimizer_state_dict' in checkpoint:
                self.rnd.optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])

        print(f"Loaded checkpoint from {filepath} (optimizers={'loaded' if load_optimizers and 'optimizer_state_dict' in checkpoint else 'not loaded'})")
