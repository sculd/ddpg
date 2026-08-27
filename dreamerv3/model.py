"""DreamerV3 networks (state-only, single-task): RSSM with 32x32 categorical
latents, symlog MLP encoder/decoder, two-hot reward head, continue head,
tanh-normal actor and two-hot critic. Follows the paper's S-size defaults
(deter 512, units 512); heads use 2 hidden layers."""
import torch
import torch.nn as nn

from dreamerv3.common import NUM_BINS, mlp


class GRUCellLN(nn.Module):
    """GRU cell with LayerNorm on the joint gate pre-activations and the -1
    update-gate bias, as in the official implementation."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, 3 * hidden_dim, bias=False)
        self.ln = nn.LayerNorm(3 * hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h):
        parts = self.ln(self.linear(torch.cat([x, h], -1)))
        reset, cand, update = parts.chunk(3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * h


class RSSM(nn.Module):
    """h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]); prior p(z_t|h_t); posterior
    q(z_t|h_t, emb_t). z is `groups` categoricals of `classes` classes with 1%
    uniform mixing and straight-through gradients, flattened to groups*classes."""

    def __init__(self, action_dim, emb_dim, deter=512, hidden=512,
                 groups=32, classes=32, unimix=0.01):
        super().__init__()
        self.deter, self.groups, self.classes = deter, groups, classes
        self.stoch = groups * classes
        self.unimix = unimix
        self.in_net = nn.Sequential(nn.Linear(self.stoch + action_dim, hidden),
                                    nn.LayerNorm(hidden), nn.SiLU())
        self.gru = GRUCellLN(hidden, deter)
        self.prior_net = mlp(deter, [hidden], self.stoch)
        self.post_net = mlp(deter + emb_dim, [hidden], self.stoch)

    def initial(self, batch, device):
        return (torch.zeros(batch, self.deter, device=device),
                torch.zeros(batch, self.stoch, device=device))

    def _core(self, h, z, a):
        return self.gru(self.in_net(torch.cat([z, a], -1)), h)

    def _logp(self, logits):
        logits = logits.view(*logits.shape[:-1], self.groups, self.classes)
        probs = (1 - self.unimix) * logits.softmax(-1) + self.unimix / self.classes
        return probs.log()

    def _sample(self, logp):
        probs = logp.exp()
        onehot = torch.distributions.OneHotCategorical(probs=probs.detach()).sample()
        st = onehot + probs - probs.detach()          # straight-through
        return st.flatten(-2)

    def obs_step(self, h, z, a_prev, emb):
        h = self._core(h, z, a_prev)
        prior_logp = self._logp(self.prior_net(h))
        post_logp = self._logp(self.post_net(torch.cat([h, emb], -1)))
        return h, self._sample(post_logp), post_logp, prior_logp

    def img_step(self, h, z, a):
        h = self._core(h, z, a)
        return h, self._sample(self._logp(self.prior_net(h)))


class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, deter=512, hidden=512,
                 groups=32, classes=32, unimix=0.01):
        super().__init__()
        self.encoder = mlp(obs_dim, [hidden], hidden)
        self.rssm = RSSM(action_dim, hidden, deter, hidden, groups, classes, unimix)
        self.feat_dim = deter + groups * classes
        self.decoder = mlp(self.feat_dim, [hidden, hidden], obs_dim)
        self.reward = mlp(self.feat_dim, [hidden, hidden], NUM_BINS, zero_out=True)
        self.cont = mlp(self.feat_dim, [hidden, hidden], 1)


class Actor(nn.Module):
    """Tanh-squashed Gaussian; std bounded in [min_std, max_std] via sigmoid."""

    def __init__(self, feat_dim, action_dim, hidden=512, min_std=0.1, max_std=1.0):
        super().__init__()
        self.net = mlp(feat_dim, [hidden, hidden], 2 * action_dim)
        self.min_std, self.max_std = min_std, max_std

    def _params(self, feat):
        mu, std_raw = self.net(feat).chunk(2, -1)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(std_raw)
        return mu, std

    def sample(self, feat):
        """Returns (tanh action, pre-tanh sample)."""
        mu, std = self._params(feat)
        u = mu + std * torch.randn_like(mu)
        return torch.tanh(u), u

    def mean_action(self, feat):
        mu, _ = self._params(feat)
        return torch.tanh(mu)

    def log_prob(self, feat, u):
        """log pi of a stored pre-tanh sample u under the current parameters."""
        mu, std = self._params(feat)
        lp = (-0.5 * ((u - mu) / std) ** 2 - std.log() - 0.9189385332046727).sum(-1)
        lp = lp - (2 * (0.6931471805599453 - u -
                        torch.nn.functional.softplus(-2 * u))).sum(-1)
        return lp

    def entropy(self, feat):
        """Entropy of the underlying Gaussian (tanh correction omitted)."""
        _, std = self._params(feat)
        return (std.log() + 0.5 * (1.0 + 1.8378770664093453)).sum(-1)


def make_critic(feat_dim, hidden=512):
    return mlp(feat_dim, [hidden, hidden], NUM_BINS, zero_out=True)
