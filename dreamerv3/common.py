"""Shared pieces of DreamerV3 (Hafner et al., 2023): symlog two-hot discrete
regression over 255 bins, Linear->LayerNorm->SiLU MLP blocks, and the EMA
percentile return normalizer. Defaults follow the paper."""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BINS = 255
VMIN, VMAX = -20.0, 20.0
BIN_SIZE = (VMAX - VMIN) / (NUM_BINS - 1)


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def twohot_encode(y):
    """y: (...,) scalar targets -> (..., NUM_BINS) two-hot weights in symlog space."""
    y = torch.clamp(symlog(y), VMIN, VMAX)
    idx = (y - VMIN) / BIN_SIZE
    lo = idx.floor().long().clamp(0, NUM_BINS - 1)
    hi = idx.ceil().long().clamp(0, NUM_BINS - 1)
    w_hi = (idx - lo.float()).clamp(0, 1)
    out = torch.zeros(*y.shape, NUM_BINS, device=y.device)
    out.scatter_(-1, lo.unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
    out.scatter_add_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


_BIN_CENTERS = None


def twohot_decode(logits):
    """logits: (..., NUM_BINS) -> (...,) scalar prediction."""
    global _BIN_CENTERS
    if _BIN_CENTERS is None or _BIN_CENTERS.device != logits.device:
        _BIN_CENTERS = torch.linspace(VMIN, VMAX, NUM_BINS, device=logits.device)
    return symexp((logits.softmax(-1) * _BIN_CENTERS).sum(-1))


def soft_ce(logits, target_scalar):
    """Cross-entropy against the two-hot encoding of a scalar target."""
    target = twohot_encode(target_scalar)
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1)


def mlp(in_dim, hidden_dims, out_dim, zero_out=False):
    """Linear -> LayerNorm -> SiLU blocks, plain Linear head. zero_out zero-inits
    the head, as the paper does for the reward predictor and critic."""
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    head = nn.Linear(d, out_dim)
    if zero_out:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


def categorical_kl(logp_q, logp_p):
    """KL(q || p) for stacks of categoricals; inputs (..., groups, classes)
    log-probs. Returns (...,) summed over groups."""
    return (logp_q.exp() * (logp_q - logp_p)).sum(-1).sum(-1)


class ReturnScale:
    """EMA of the 5th and 95th percentiles of imagined returns; advantages are
    divided by max(1, spread) as in the paper."""

    def __init__(self, decay=0.99):
        self.decay = decay
        self.p5 = None
        self.p95 = None

    def update(self, x):
        p5, p95 = torch.quantile(x.detach().float().flatten(),
                                 torch.tensor([0.05, 0.95], device=x.device))
        if self.p5 is None:
            self.p5, self.p95 = float(p5), float(p95)
        else:
            self.p5 = self.decay * self.p5 + (1 - self.decay) * float(p5)
            self.p95 = self.decay * self.p95 + (1 - self.decay) * float(p95)

    @property
    def scale(self):
        if self.p5 is None:
            return 1.0
        return max(1.0, self.p95 - self.p5)
