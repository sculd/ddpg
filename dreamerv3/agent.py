"""DreamerV3 agent (Hafner et al., 2023): world model trained by reconstruction
+ reward/continue prediction + KL balancing with free bits on replayed
sequences; actor-critic trained purely on imagined latent rollouts with
lambda-returns, percentile return normalization and REINFORCE gradients.
No planning at act time - the policy acts directly from the posterior state.

Simplifications vs the official code: state-only MLP encoder/decoder,
within-episode sequence sampling (cold-start h=0 at window starts, which the
official code also does via is_first masking), tanh-normal actor."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from dreamerv3.buffer import SeqReplayBuffer
from dreamerv3.common import ReturnScale, categorical_kl, soft_ce, symlog, twohot_decode
from dreamerv3.model import Actor, WorldModel, make_critic
from sac.noise import ColoredNoiseProcess


def lambda_return(reward, cont, value, gamma, lam):
    """reward, cont: (H, N) predicted at imagined states 1..H; value: (H+1, N).
    Returns (H, N) lambda-return targets for states 0..H-1."""
    R = value[-1]
    outs = []
    for t in reversed(range(reward.shape[0])):
        R = reward[t] + gamma * cont[t] * ((1 - lam) * value[t + 1] + lam * R)
        outs.append(R)
    return torch.stack(outs[::-1])


class DreamerV3:
    def __init__(self, obs_dim, action_dim,
                 deter=512, hidden=512, groups=32, classes=32, unimix=0.01,
                 seq_len=32, batch_size=16, imag_horizon=15,
                 gamma=0.997, lam=0.95,
                 model_lr=1e-4, ac_lr=3e-5,
                 dyn_coef=0.5, rep_coef=0.1, free_bits=1.0,
                 entropy_coef=3e-4, slow_tau=0.02,
                 model_clip=1000.0, ac_clip=100.0,
                 buffer_capacity=1_000_000, device=None,
                 noise_beta=0.0, noise_seq_len=1000, collect_min_std=0.0):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.seq_len, self.batch_size, self.imag_horizon = seq_len, batch_size, imag_horizon
        self.gamma, self.lam = gamma, lam
        self.dyn_coef, self.rep_coef, self.free_bits = dyn_coef, rep_coef, free_bits
        self.entropy_coef, self.slow_tau = entropy_coef, slow_tau
        self.model_clip, self.ac_clip = model_clip, ac_clip

        self.model = WorldModel(obs_dim, action_dim, deter, hidden,
                                groups, classes, unimix).to(self.device)
        self.actor = Actor(self.model.feat_dim, action_dim, hidden).to(self.device)
        self.critic = make_critic(self.model.feat_dim, hidden).to(self.device)
        self.critic_slow = copy.deepcopy(self.critic)
        for p in self.critic_slow.parameters():
            p.requires_grad_(False)
        self.model_optim = torch.optim.Adam(self.model.parameters(), lr=model_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=ac_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=ac_lr)
        self.ret_scale = ReturnScale()
        self.buffer = SeqReplayBuffer(buffer_capacity, obs_dim, action_dim, seq_len)
        # colored collection noise (Eberhard et al., ICLR 2023): replaces the
        # i.i.d. eps in u = mu + std*eps at act time only; beta=0 is unchanged
        self.noise = (ColoredNoiseProcess(noise_beta, action_dim, noise_seq_len)
                      if noise_beta > 0 else None)
        # collection-only exploration-std floor: keeps behaviour-policy amplitude
        # from collapsing with the learned std; training and eval are untouched
        self.collect_min_std = collect_min_std
        self.reset_episode()

    # ---------------- acting ----------------
    def reset_episode(self):
        self._h = None
        self._z = None
        self._a = None
        if self.noise is not None:
            self.noise.reset()

    @torch.no_grad()
    def act(self, obs, eval_mode=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self._h is None:
            self._h, self._z = self.model.rssm.initial(1, self.device)
            self._a = torch.zeros(1, self.action_dim, device=self.device)
        emb = self.model.encoder(symlog(obs))
        self._h, self._z, _, _ = self.model.rssm.obs_step(self._h, self._z, self._a, emb)
        feat = torch.cat([self._h, self._z], -1)
        if eval_mode:
            a = self.actor.mean_action(feat)
        elif self.noise is not None or self.collect_min_std > 0:
            mu, std = self.actor._params(feat)
            if self.collect_min_std > 0:
                std = std.clamp_min(self.collect_min_std)
            if self.noise is not None:
                eps = torch.as_tensor(self.noise.sample(), device=self.device).unsqueeze(0)
            else:
                eps = torch.randn_like(mu)
            a = torch.tanh(mu + std * eps)
        else:
            a, _ = self.actor.sample(feat)
        self._a = a
        return a[0].cpu().numpy()

    # ---------------- learning ----------------
    def update(self):
        obs, action, reward, term = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(obs, device=self.device)         # (L+1, B, obs)
        action = torch.as_tensor(action, device=self.device)   # (L, B, act)
        reward = torch.as_tensor(reward, device=self.device)   # (L, B)
        term = torch.as_tensor(term, device=self.device)       # (L, B)
        L1, B = obs.shape[0], obs.shape[1]

        # ---- world model: posterior rollout over the sequence ----
        emb = self.model.encoder(symlog(obs))
        prev_a = torch.cat([torch.zeros_like(action[:1]), action], 0)  # a_{t-1}
        h, z = self.model.rssm.initial(B, self.device)
        hs, zs, post_lps, prior_lps = [], [], [], []
        for t in range(L1):
            h, z, post_lp, prior_lp = self.model.rssm.obs_step(h, z, prev_a[t], emb[t])
            hs.append(h); zs.append(z)
            post_lps.append(post_lp); prior_lps.append(prior_lp)
        h_seq, z_seq = torch.stack(hs), torch.stack(zs)
        feat = torch.cat([h_seq, z_seq], -1)                   # (L+1, B, F)
        post_lp, prior_lp = torch.stack(post_lps), torch.stack(prior_lps)

        obs_loss = ((self.model.decoder(feat) - symlog(obs)) ** 2).sum(-1).mean()
        # reward/continue heads predict at state t the outcome of the step into t
        reward_loss = soft_ce(self.model.reward(feat[1:]), reward).mean()
        cont_loss = F.binary_cross_entropy_with_logits(
            self.model.cont(feat[1:]).squeeze(-1), 1.0 - term)
        dyn_loss = categorical_kl(post_lp.detach(), prior_lp).clamp_min(self.free_bits).mean()
        rep_loss = categorical_kl(post_lp, prior_lp.detach()).clamp_min(self.free_bits).mean()
        model_loss = (obs_loss + reward_loss + cont_loss +
                      self.dyn_coef * dyn_loss + self.rep_coef * rep_loss)
        self.model_optim.zero_grad(set_to_none=True)
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_clip)
        self.model_optim.step()

        # ---- imagination from every posterior state (detached) ----
        H = self.imag_horizon
        with torch.no_grad():
            start_cont = torch.cat([torch.ones_like(term[:1]), 1.0 - term], 0).reshape(-1)
            h = h_seq.reshape(-1, h_seq.shape[-1])
            z = z_seq.reshape(-1, z_seq.shape[-1])
            feats, us = [torch.cat([h, z], -1)], []
            for _ in range(H):
                a, u = self.actor.sample(feats[-1])
                us.append(u)
                h, z = self.model.rssm.img_step(h, z, a)
                feats.append(torch.cat([h, z], -1))
            feats, us = torch.stack(feats), torch.stack(us)    # (H+1,N,F), (H,N,A)
            r = twohot_decode(self.model.reward(feats[1:]))           # (H, N)
            c = torch.sigmoid(self.model.cont(feats[1:]).squeeze(-1))  # (H, N)
            v = twohot_decode(self.critic(feats))                      # (H+1, N)
            ret = lambda_return(r, c, v, self.gamma, self.lam)         # (H, N)
            w = [start_cont]
            for t in range(H - 1):
                w.append(w[-1] * self.gamma * c[t])
            w = torch.stack(w)                                         # (H, N)
            self.ret_scale.update(ret)
            adv = (ret - v[:-1]) / self.ret_scale.scale
            slow_v = twohot_decode(self.critic_slow(feats[:-1]))

        logpi = self.actor.log_prob(feats[:-1], us)
        entropy = self.actor.entropy(feats[:-1])
        actor_loss = -(w * (adv * logpi + self.entropy_coef * entropy)).mean()
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.ac_clip)
        self.actor_optim.step()

        critic_logits = self.critic(feats[:-1])
        critic_loss = (w * (soft_ce(critic_logits, ret) +
                            soft_ce(critic_logits, slow_v))).mean()
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.ac_clip)
        self.critic_optim.step()
        with torch.no_grad():
            for p, sp in zip(self.critic.parameters(), self.critic_slow.parameters()):
                sp.lerp_(p, self.slow_tau)

        with torch.no_grad():
            # stall diagnostic: has the reward head seen anything above the floor?
            reward_max = twohot_decode(self.model.reward(feat[1:])).max().item()
        return {'model': model_loss.item(), 'obs': obs_loss.item(),
                'kl': dyn_loss.item(), 'reward': reward_loss.item(),
                'actor': actor_loss.item(), 'critic': critic_loss.item(),
                'ret': ret.mean().item(), 'rmax': reward_max}

    # ---------------- io ----------------
    def save(self, path):
        torch.save({'model': self.model.state_dict(),
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'critic_slow': self.critic_slow.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model'])
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_slow.load_state_dict(ckpt['critic_slow'])
