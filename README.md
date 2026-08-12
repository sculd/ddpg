# Reinforcement Learning On Bipedal Walker

* DDPG: Deep Deterministic Policy Gradient 
  * https://arxiv.org/abs/1509.02971
* SAC: Soft Actor Critic
  * https://arxiv.org/abs/1801.01290
* FORK: Foward Looking
  * https://arxiv.org/abs/2010.01652
* CN = Colored (Pink) Noise exploration
  * https://arxiv.org/abs/2206.05403

Env: `BipedalWalker-v3`, and its `BipedalWalkerHardcore-v3` variant (much harder).

## Result
#### DDPG
DDPG lacks the exploration capacity, thus it takes several runs to achieve the learning curve that is on the right trajectory.

<img src="images/LunarLander.png" width="100%" height="50%">

<img src="images/animation_dupg.gif" width="50%" height="50%">

#### SAC (Easy)
SAC augments DDPG with entropy term in the score thus it can achieve learning the non-hardcore environment within single run. But it lacks the capacity to learn the hardcore environment, mainly because it lacks the ability to handle the big stump obstacle which requires more forward looking planning.

<img src="images/episode_reward_sac_nonhardcore.png" width="50%" height="50%">

<img src="images/animation_sac_nonhardcore.gif" width="50%" height="50%">

#### SAC-FORK (Hardcore)
SAC-FORK augments SAC by adding forward looking term. This allwos it to achieve learning the hardcore environment after ~10M steps.

<img src="images/episode_reward_sac_fork_hardcore.png" width="50%" height="50%">

<img src="images/animation_sac_fork_hardcore.gif" width="50%" height="50%">

## MountainCarContinuous-v0 (sparse reward)

Plain SAC and SAC-FORK both fail here (evaluated at -0.3 and -67 respectively): the
action penalty makes "do nothing" a stable local optimum, and per-step i.i.d. Gaussian
noise is a random walk in torque that never builds the momentum needed to reach the goal.

SAC-CN (`sac/agent_cn.py`) keeps the SAC update rules unchanged and only replaces the
exploration noise with a temporally correlated 1/f^beta (pink, beta=1) sequence per
episode, following "Pink Noise Is All You Need" (Eberhard et al., ICLR 2023). The
correlated torques produce coherent rocking behavior: the very first exploration episode
already reaches the goal.

Result: **94.7 +/- 0.4 average over 100 deterministic eval episodes** (solved bar: 90),
goal reached in ~70 steps, after ~300k env steps (~13 min wall clock).

```
$ python train_sac.py --config-name=train_cn_mountain_car.yaml
$ python test_sac.py --config-name=test_cn_mountain_car.yaml
```

`noise_beta: 0` in `configs_sac/agent/sac_cn_mountain_car.yaml` recovers plain SAC;
`2.0` gives OU-like red noise.

Note: gymnasium >= 1.0 vector envs auto-reset in "next step" mode; the training loop
skips the bookkeeping transition at episode boundaries and bootstraps through time-limit
truncations (done flag = terminated only).

## Batchsize
Note: `Small batch deep reinforcement learning` [1509.02971](https://arxiv.org/abs/1509.02971), suggests a smaller batch size of 16, but my observation does not align with it.

## Mountain Car
Mountain Car is hard because the agent must perform a long sequence of seemingly counterproductive actions—moving away from the goal to build enough momentum—before receiving the sparse reward at the top of the hill.

<img src="images/animation_sac_mountain_car_fail.gif" width="50%" height="50%">

(See how the agent learns to minimize the car movement to avoid the penalty that comes just from moving).

* RND: Random Network Distillation
  * https://arxiv.org/abs/1810.12894

Exploratory algorithms like SAC and RND fail to solve the Mountain Car problem because they rewards novel states rather than goal-directed progress. In other words, they drive curiosity but not purposeful momentum accumulation.

#### SAC-CN (solved)
SAC with pink-noise exploration (see the MountainCarContinuous section above) solves it:
the temporally correlated noise produces the coherent rocking that builds momentum. The
trained agent reaches the flag in ~70 steps, scoring ~95 per episode.

<img src="images/animation_sac_cn_mountain_car.gif" width="50%" height="50%">

## Environments

* Pytorch >= 2.5.1

## Profiling
`train_sac.py` has profiling implemented. After running with `--profile` flag, run
```
$ snakeviz profiles/training_profile.prof
$ tensorboard --logdir=./profiles/torch_profilerer
```

