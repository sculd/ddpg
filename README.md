# Reinforcement Learning On Bipedal Walker

DDPG = Deep Deterministic Policy Gradient
SAC = Soft Actor Critic
FORK = Foward Looking


* DDPG: https://arxiv.org/abs/1509.02971
* SAC: https://arxiv.org/abs/1801.01290
* FORK: https://arxiv.org/abs/2010.01652

Env: `BipedalWalker-v3`, and its `BipedalWalkerHardcore-v3` variant (much harder).

## Result
#### DDPG

<img src="images/LunarLander.png" width="100%" height="50%">

<img src="images/animation_dupg.gif" width="50%" height="50%">

#### SAC (Easy)

<img src="images/episode_reward_sac_nonhardcore.png" width="50%" height="50%">

#### SAC-FORK (Hardcore)

<img src="images/episode_reward_sac_fork_hardcore.png" width="50%" height="50%">

## Batchsize
Note: `Small batch deep reinforcement learning` [1509.02971](https://arxiv.org/abs/1509.02971), suggests a smaller batch size of 16, but my observation does not align with it.


## Environments

* Pytorch >= 2.5.1
