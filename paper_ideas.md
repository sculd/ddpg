# Paper ideas

Research directions that an individual with one GPU and this codebase can take to a
publishable result. Written 2026-08-18 after the HER / FetchReach investigation
(see the README section "FetchReach (sparse, goal-conditioned)" and `her/`).

## What we already know (evidence from this repo)

- **MountainCarContinuous**: SAC and SAC-FORK fail; SAC with pink-noise exploration
  (beta = 1) solves it. Correlated action noise is decisive when the behaviour needed
  to *ever* see reward is temporally extended (rocking to build momentum).
- **FetchReach-v4**: with paper-faithful DDPG+HER the colour of the exploration noise is
  irrelevant (white / pink / red / OU all reach 100 % within 2-5k env steps, 3 seeds);
  without HER it is still irrelevant (all solve by ~15k steps). A random policy already
  touches the goal in ~18 % of episodes, so exploration is not the bottleneck. The
  earlier failure was optimisation (goal dropped from the learner, lr 1e-5, tau 1e-3,
  no input scaling, no action-L2, buffer wiped per epoch).
- Single-fix ablation on the broken config: input normalisation alone, or lr 1e-3
  alone, or action-L2 alone rescues it; more exploration (30 % random actions) does
  nothing. Noise colour and HER operate on different bottlenecks.

These two anchor points (colour decisive vs. colour irrelevant) define a boundary that
nobody has mapped. That boundary is Idea 1.

## Literature check (2026-08-18)

The colored-noise-exploration line is three papers:

1. Eberhard et al., *Pink Noise Is All You Need*, ICLR 2023 (oral) - off-policy
   SAC/MPO on DMC / MetaWorld-style tasks; recommends beta = 1. No goal-conditioned or
   Fetch tasks. https://openreview.net/forum?id=hQ9V5QN27eS ,
   code https://github.com/martius-lab/pink-noise-rl
2. Hollenstein et al., *Colored Noise in PPO*, AAAI 2024 - on-policy; beta ~ 0.5 best.
   https://arxiv.org/abs/2312.11091
3. *A Temporally Correlated Latent Exploration for RL*, Dec 2024 - correlated noise in
   the latent / intrinsic-motivation path (partially covers Idea 2 below).
   https://arxiv.org/pdf/2412.04775

Nothing on colored noise x hindsight relabelling / goal-conditioned RL / sparse-reward
manipulation. Re-check (arXiv, OpenReview, Semantic Scholar) before submitting.

Related HER-side work to cite: Andrychowicz et al. 2017 (HER); Plappert et al. 2018
(Fetch/Hand benchmark, the baselines hyper-parameters); Pitis et al. 2020 (MEGA, goal
selection by achieved-goal density - closest in spirit to the coverage predictor);
GCHR (arXiv 2508.06108, Aug 2025, hindsight regularisation); DISCOVER (arXiv 2505.19850,
automated curricula for sparse reward, cites pink noise).

---

## Idea 1 (primary): Does exploration-noise colour matter for goal-conditioned RL?

Working titles: *"Does Exploration Noise Colour Matter for Goal-Conditioned RL?"*;
if the effect is positive on the harder tasks, *"Pink Noise Meets Hindsight"*.

### Question
Hindsight relabelling and temporally correlated exploration both attack sparse reward.
Are they substitutes, complements, or orthogonal - and can we predict, per task, when
noise colour will matter?

### Hypotheses
- H1: colour matters exactly on tasks where beta > 0 materially expands the
  *achieved-goal coverage* of an untrained/early policy (Push, Slide: sustained
  directional contact needed; MountainCar: momentum). On tasks where white noise
  already covers the goal space (Reach), neither colour nor HER changes much.
- H2: HER converts coverage into learning signal, so the gain from colour is larger
  *with* HER than without on the coverage-limited tasks (complements), and the gain
  from HER is larger with pink than with white noise.
- H3 (secondary): the best beta for goal-conditioned DDPG is ~1, as in the off-policy
  paper, not ~0.5 as in the on-policy paper.

### Experimental grid
- Noise: beta in {0 (white), 0.5, 1 (pink), 2 (red)}; OU as the historical baseline.
- HER: on (future, k = 4) / off.
- Tasks: FetchReach, FetchPush, FetchPickAndPlace, FetchSlide (gymnasium-robotics v4);
  2-3 non-goal-conditioned sparse controls (MountainCarContinuous; sparse DMC or
  MetaWorld tasks) to connect to the 2023 paper.
- Algorithms: DDPG+HER (`her/`), and SAC+HER as a second backbone to show the effect
  is not DDPG-specific (reuse `sac/agent_cn.py` noise plumbing).
- Seeds: 10 per cell (5 minimum for the controls).
- Size: ~4 beta x 2 HER x 4 Fetch x 10 seeds = 320 DDPG runs + controls + SAC repeat
  ~ 600-800 runs. Reach is minutes; Push/PickAndPlace ~1-2 h each on one GPU
  (CPU-bound MuJoCo; run 8-12 in parallel as in `exp_her/jobs*.txt`). A few GPU-weeks
  total; start with Push only.

### The predictor (what lifts it from ablation to paper)
Before any training, measure per task and per beta the **achieved-goal coverage**
of a fixed exploration policy (random or a freshly initialised actor + noise):
entropy / number of occupied bins of the achieved-goal distribution over N episodes,
or mean nearest-neighbour distance from sampled desired goals to achieved goals.
Cheap (no learning). Claim to test: the *training-time* benefit of beta is predicted by
the *coverage* gain of beta. Report the rank correlation across tasks. This is the
figure that makes the paper.

Secondary diagnostics: HER relabel hit rate (fraction of relabelled transitions with
reward 0), goal-space frontier expansion over training (MEGA-style density), steps to
first success.

### The MountainCar 2x2 (proposed 2026-08-18)
Make MountainCarContinuous goal-conditioned: achieved_goal = car position (1-D),
reward = 0 if |x - g| < 0.05 else -1, drop the env's action penalty; train with the
flag (x = 0.45) as desired goal, evaluate on the flag and on uniformly sampled g.
This completes the 2x2 {white, pink} x {HER, no HER} on the most diagnostic task:
no-HER row is already measured (white fails, pink solves).
- HER+white solves it -> HER *substitutes* for correlated noise (relabelling extracts
  a curriculum from uncorrelated wiggling; the achieved-goal frontier is the energy
  frontier, so HER's curriculum aligns with the task physics).
- HER+white fails, HER+pink >= pink alone -> *complements*; the coverage probe should
  show white noise's achieved-goal distribution collapsing near the start state.
Either outcome feeds H1/H2. The 1-D goal space makes the coverage probe trivial and
gives the best figure of the paper: achieved-goal histogram vs training time per
(beta, HER) cell. Known failure mode to discuss: uniform `future` relabelling
concentrates on well-visited goals (the MEGA/Skew-Fit critique); check GCSL /
Skew-Fit / MEGA for prior goal-conditioned MountainCar variants before claiming
novelty of the cell itself.

**Measured (2026-08-18, DDPG+HER `her/`, T=200, 30 epochs = 1.2M steps, 3 seeds,
`--set noise_eps=1.0 random_eps=0.0 action_l2=0.0 gamma=0.995`).**
Env wrapper: `her/mountain_car.py` (GoalMountainCar-v0); runs in `exp_her/mc_*.csv`,
job lists `exp_her/jobs_mc*.txt`.

| solved (final eval >= 0.9) | white | pink |
|---|---|---|
| no HER | 2/3 (first solve ep 10, 13) | 1/3 (ep 5) |
| + HER  | 3/3 (ep 10, 24, 25)        | 3/3 (ep 4-23, median ~7) |

- Headline: with correct hyper-parameters, **HER makes the task reliably solvable
  with either noise colour** (substitution), while **either ingredient alone is
  unreliable** (1-3 of 3). Pink+HER solves ~3x earlier (median first-solve epoch ~7
  vs ~24) - colour buys speed, HER buys reliability. More nuanced than pure
  substitutes or complements; n=3, so re-run with 10 seeds before claiming rates.
- White+HER succeeding is the notable cell: train_reached starts at 0 (white noise
  never touches the flag early) yet the relabelling curriculum bootstraps to 100 %.
- **Pitfall that cost a day: the Fetch-preset `action_l2=1.0` re-creates the
  "do nothing" optimum** this env is famous for (we removed the env's action penalty
  and re-added one via the actor loss). With action_l2=1.0 every cell is 0/3 - even
  pink+HER with 10-25 % of training episodes touching the flag. Removing it (fixA)
  alone gives 3/3; gamma 0.98->0.995 alone (fixG) gives 0/3 and is not required at
  T=200. Weak exploration (noise_eps=0.2 + 30 % random actions) also fails everywhere.
  This hyper-parameter x task-structure interaction is itself paper material.
- Deviation from the SAC-CN anchor: pink *alone* solved MountainCar reliably with SAC
  (T=1000, learned temperature); with DDPG at T=200 pink alone is 1/3. Episode length
  and backbone matter; the paper grid should include T in {200, 1000}.

**Main-branch reproduction (2026-08-20, `train_her_mountain_car.py` +
`envs/goal_mountain_car_env.py`, same hyper-parameters but the repo's own DDPG
implementation: BatchNorm 400/300 nets, cosine LR cycling, ~100 updates/episode
i.e. ~5x the her_fix schedule, episodes break on success, no deterministic eval -
success rates below are training-time).** 8 epochs x 800 episodes (~1.3M steps):
- white + HER: 41 % -> 95 % by epoch 2, stays 82-99 %. Solved.
- pink + HER: seed 0 oscillates 4-38 % and never converges; seed 1 collapses to
  <1 % after epoch 1. 0/2.
The headline "HER makes the task learnable" reproduces, but the colour ordering
*flips* with the training regime: on the her_fix trainer pink+HER was fastest and
white+HER slowest; on main's heavier update schedule white+HER is strong and
pink+HER is unstable. Which noise colour wins is not a property of the task alone
but of the (task, backbone, update schedule) triple - a caution for the paper's
claims, and worth a dedicated ablation (updates-per-episode x noise colour).

### Metrics / reporting
- Eval success rate (paper protocol: 10 deterministic episodes per epoch), steps to
  reach 90 % / 100 %.
- rliable: IQM + stratified bootstrap CIs, performance profiles; per-task and
  aggregated over the Fetch suite.
- Release code + all CSVs (the `exp_her/` harness and `her/plot_results.py` are the
  skeleton).

### Venues
RLC (Reinforcement Learning Conference) first choice; TMLR (rolling, correctness over
hype); NeurIPS / ICLR workshop for early feedback; arXiv preprint + repo as the resume
artefact in any case.

### Risks and insurance
- Null result ("colour never matters once HER is on"): still publishable *if* the
  coverage predictor explains it. Do the predictor first.
- Scooping: the obvious "pink noise + HER on Fetch" table could appear any time; the
  predictor + Fetch-suite focus + two backbones differentiate. Aim for a preprint in
  3-6 months.
- PickAndPlace may need the paper's trick (start half the episodes with the object in
  the gripper) or it stalls for everyone; report it as-is either way.

### Next steps
0. Run the MountainCar 2x2 first: cheapest experiment, most informative per run
   (wrap MountainCarContinuous in a goal-conditioned interface compatible with
   `her/replay_buffer.py`'s reward_fn; ~30 lines).
1. Make `her/` run Push / PickAndPlace / Slide (loop is env-agnostic; add the
   gripper-start option; check `compute_reward` vectorisation).
2. Implement the coverage probe (~50 lines; reuse `her/agent.py::ExplorationNoise`).
3. Launch the Push grid (beta x HER x 10 seeds) first - most likely to show an effect.
4. Add SAC+HER backbone; then the remaining tasks and controls.
5. Write as you go: the README section is already the related-work + method draft.

---

## Idea 2: Temporally correlated intrinsic motivation

**Question.** Pink action noise produces coherent exploration; RND-style bonuses do
not. Does injecting temporal correlation into the *intrinsic* signal (a slowly varying
perturbation of the RND predictor target, the bonus scale, or the latent used for
novelty) give the same coherence benefit? Testbed: MountainCar (where RND failed in
this repo) and the Fetch tasks without HER.

**Status.** Partially touched by the Dec 2024 "temporally correlated latent
exploration" paper - read it first; position as the intrinsic-reward (not latent)
variant or as a comparison. Higher novelty, higher risk than Idea 1. Reuses
`rnd/` + `sac/noise.py`.

## Idea 3: "What matters in HER" - a rigorous ablation / reproduction study

**Question.** Which of the HER-paper details are load-bearing, on which tasks, and
how do they interact? Today's single-fix result (normalisation, lr, action-L2 each
sufficient on Reach; batch, gamma, target clipping, extra exploration not) is the seed.
Extend to Push/PickAndPlace; factors: input normalisation, action-L2, k in {1,2,4,8},
future/episode/final, polyak, target-Q clipping, target-actor-for-eval, updates per
episode, exploration scheme (20 %/5 % paper vs 30 %/0.2 baselines vs OU).

**Why publishable.** Modelled on Andrychowicz et al. 2020 *What Matters in On-Policy
RL* (high-impact, engineering-only). Cheapest idea here; lower ceiling; good TMLR /
RLC / reproducibility-track fit. Pairs naturally with Idea 1 (same runs, different
axes).

## Idea 4: Colored noise under massive parallelism

**Question.** FastTD3 / SimbaV2-style training uses thousands of parallel envs; does
per-env temporally correlated noise still help when diversity already comes from
parallelism, or does the benefit vanish (or invert)? MJX / Isaac Lab on one GPU;
FastTD3 code is small. Timely; clean yes/no; connects Idea 1 to the 2025 state of the
art. Reference: FastTD3 https://arxiv.org/abs/2505.22642

## Idea 5: Cheap real robot - HER / HIL-SERL on an SO-100 arm

**Question.** On a ~$200 arm with the LeRobot stack, does pink noise or hindsight
relabelling shorten human-in-the-loop RL time (HIL-SERL-style) on reach / pick tasks?
Real-robot results carry weight far beyond their cost; hardware and patience are the
barrier, not compute.

---

## General guidance

- 5-10 seeds, rliable CIs, pre-registered hypotheses (write them down before the grid
  runs), negative results reported as first-class.
- Publish the harness and raw CSVs; reviewers at RLC/TMLR reward reproducibility.
- Re-run the literature check right before submission; cite the three colored-noise
  papers and the HER/Fetch lineage.
- Target order: arXiv preprint -> workshop feedback -> RLC or TMLR.
