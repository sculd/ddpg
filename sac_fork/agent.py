import copy

import numpy as np
import torch
import torch.nn.functional as F

from sac.network import DiagGaussianActor, DoubleQCritic
from sac_fork.network import Sys_R, SysModel
from sac_fork.utils import hard_update, soft_update


class SAC_FORK(object):
    def __init__(self, obs_dim, action_space, action_range, device, args, **kwargs):
        self.action_range = action_range

        self.device = torch.device(device)
        self.mode = True # training mode
        self.gamma = args.gamma
        self.tau = args.tau
        self.log_alpha = torch.tensor(np.log(args.alpha)).to(self.device)
        self.log_alpha.requires_grad = True

        self.policy_type = args.policy_type
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        action_dim = action_space.shape[0]
        self.critic = DoubleQCritic(obs_dim, action_dim, args.hidden_size, 3).to(device=self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = DoubleQCritic(obs_dim, action_dim, args.hidden_size, 3).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.sysmodel = SysModel(obs_dim, action_dim, args.sys_hidden_size,args.sys_hidden_size).to(self.device)
        self.sysmodel_optimizer = torch.optim.Adam(self.sysmodel.parameters(), lr=args.lr)

        # state space upper/lower bound
        self.obs_upper_bound = 10
        self.obs_lower_bound = -10

        self.sysr = Sys_R(obs_dim, action_dim,args.sysr_hidden_size,args.sysr_hidden_size).to(self.device)
        self.sysr_optimizer = torch.optim.Adam(self.sysr.parameters(), lr=args.lr)

        self.sys_threshold = args.sys_threshold
        self.sys_weight = args.sys_weight
        self.sysmodel_loss = 0
        self.sysr_loss = 0

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            # set target entropy to -|A|
            self.target_entropy = -action_dim
            self.log_alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

        self.actor = DiagGaussianActor(obs_dim, action_dim, args.hidden_size, hidden_depth=3).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def reset(self):
        pass

    def train(self, mode=True):
        # mode False means eval mode
        self.mode = mode
        if mode:
            self.sysr.train()
            self.sysmodel.train()
            self.actor.train()
            self.critic.train()
        else:
            self.sysr.eval()
            self.sysmodel.eval()
            self.actor.eval()
            self.critic.eval()

    def act(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        # Handle both single observations and batches
        if state.ndim == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        dist = self.actor(state)
        action = dist.sample() if not evaluate else dist.mean
        action = action.clamp(*self.action_range)

        action = action.detach().cpu().numpy()
        if squeeze_output:
            return action[0]
        else:
            return action

    def update_critic(self, states, actions, rewards, next_states, masks, logger, step):
        dist = self.actor(next_states)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_states, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = rewards + (masks * self.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.critic.log(logger, step)


    def update_actor(self, states, actions, rewards, next_states, logger,step):
        dist = self.actor(states)
        action_sampled = dist.rsample()
        log_prob = dist.log_prob(action_sampled).sum(-1, keepdim=True)
        critic_Q1, critic_Q2 = self.critic(states, action_sampled)

        critic_Q = torch.min(critic_Q1, critic_Q2)
        actor_loss = -(critic_Q - self.alpha.detach() * log_prob).mean()

        # fork logic
        predict_next_state = self.sysmodel(states, actions)
        predict_next_state = predict_next_state.clamp(self.obs_lower_bound, self.obs_upper_bound)
        sysmodel_loss = F.smooth_l1_loss(predict_next_state, next_states.detach())
        self.sysmodel_optimizer.zero_grad()
        sysmodel_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sysmodel.parameters(), max_norm=1.0)
        self.sysmodel_optimizer.step()
        self.sysmodel_loss = sysmodel_loss.item()

        predict_reward = self.sysr(states, next_states, actions)
        sysr_loss = F.mse_loss(predict_reward, rewards.detach())
        self.sysr_optimizer.zero_grad()
        sysr_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sysr.parameters(), max_norm=1.0)
        self.sysr_optimizer.step()
        self.sysr_loss = sysr_loss.item()

        s_flag = 1 if sysmodel_loss.item() < self.sys_threshold else 0

        if s_flag == 1 and self.sys_weight != 0:
            next_state_system = self.sysmodel(states, action_sampled).clamp(self.obs_lower_bound, self.obs_upper_bound)
            r_sysr = self.sysr(states, next_state_system.detach(), action_sampled)
            next_dist = self.actor(next_state_system.detach())
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)

            next2_state_system = self.sysmodel(next_state_system, next_action).clamp(self.obs_lower_bound, self.obs_upper_bound)
            r_next_sysr = self.sysr(next_state_system.detach(), next2_state_system.detach(), next_action)
            next2_dist = self.actor(next2_state_system.detach())
            next2_action = next2_dist.rsample()
            next2_log_prob = next2_dist.log_prob(next2_action).sum(-1, keepdim=True)

            '''
            2-step rollout with the system model, then bootstrapping with the critic network:
            - Steps 0→1: use system model (sysmodel + sysr)
            - Steps 1→2: use system model (sysmodel + sysr)
            - Steps 2+: bootstrap with critic network

            Using the critic here provides a proper Q-value estimate for the continuation,
            rather than relying on the system model indefinitely (which would accumulate errors).            
            '''
            critic_next2_Q1, critic_next2_Q2 = self.critic(next2_state_system.detach(), next2_action)
            critic_next2_Q = torch.min(critic_next2_Q1, critic_next2_Q2)

            '''
            Question: The vanilla sac version's alpha * log_prob is paired with Q value, not reward. 
            Q ~ reward + gamma * Q, where gamma ~0.99 thus Q is of much larger scale than reward (about 100 times). 
            Yet, the sys_loss term has r_sysr - self.alpha * log_prob, pairing of r, not q.
            '''
            sys_loss = -((r_sysr - self.alpha * log_prob) + self.gamma * (r_next_sysr - self.alpha * next_log_prob) + self.gamma ** 2 * (critic_next2_Q - self.alpha * next2_log_prob)).mean()
            actor_loss += self.sys_weight * sys_loss

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)
        logger.log('train_sysmodel/sysmodel_loss', sysmodel_loss.item(), step)
        logger.log('train_sysmodel/s_flag', s_flag, step)
        logger.log('train_sysr/sysr_loss', sysr_loss.item(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.automatic_entropy_tuning:
            self.log_alpha_optim.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.log_alpha_optim.step()

        if step % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

    
    def update(self, replay_buffer, batch_size, logger, step):
        states, actions, rewards, next_states, masks = replay_buffer.sample(batch_size=batch_size)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)

        logger.log('train/batch_reward', rewards.mean(), step)

        self.update_critic(states, actions, rewards, next_states, masks, logger, step)
        self.update_actor(states, actions, rewards, next_states, logger, step)


    # Save model parameters
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.sysmodel.state_dict(), filename + "_sysmodel")
        torch.save(self.sysmodel_optimizer.state_dict(), filename + "_sysmodel_optimizer")

        torch.save(self.sysr.state_dict(), filename + "_reward_model")
        torch.save(self.sysr_optimizer.state_dict(), filename + "_reward_model_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.sysmodel.load_state_dict(torch.load(filename + "_sysmodel"))
        self.sysmodel_optimizer.load_state_dict(torch.load(filename + "_sysmodel_optimizer"))

        self.sysr.load_state_dict(torch.load(filename + "_reward_model"))
        self.sysr_optimizer.load_state_dict(torch.load(filename + "_reward_model_optimizer"))
