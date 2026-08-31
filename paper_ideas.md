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

### TD-MPC2 backbone (measured 2026-08-21)
State-only TD-MPC2 reimplementation in `tdmpc2/` (paper defaults: SimNorm latents,
symlog two-hot reward/Q heads, 5-Q ensemble, MPPI with policy prior, horizon 3),
verified on the ladder:
- Pendulum: eval return -191 +/- 69 (solved; validates the implementation).
- FetchReach-v4 (goal flattened into obs, no HER): **100 % eval success by ~15k
  steps, both seeds** - planning through the learned model handles the easy sparse
  task without relabelling.
- MountainCarContinuous: **fails exactly like model-free SAC** - returns ~ -0.2
  (the "do nothing" optimum: it minimises the action penalty and never sees the
  goal reward, so imagination has nothing to plan toward). Model-based planning
  does NOT substitute for exploration; the model cannot conjure reward it has
  never observed.
This is the third anchor for H1 and tees up the open experiment: colored noise in
the planner. Note the paper-grid subtlety - TD-MPC2's horizon is 3, so in-horizon
colored MPPI sampling (the iCEM trick, Pinneri et al. 2020) is nearly meaningless;
the correlation has to go into the *executed* action noise across env steps
(colour the std-noise applied to the first planned action), and/or lengthen the
planning horizon. Both are ~20-line changes in `tdmpc2/agent.py::act`.

### DreamerV3 backbone (measured 2026-08-27/28)
State-only DreamerV3 reimplementation in `dreamerv3/` (paper defaults: RSSM with
32x32 categorical latents, symlog two-hot reward/critic heads, KL balancing with
free bits, imagination-trained actor-critic, REINFORCE with percentile return
normalisation; no planning at act time), verified on the ladder
(curves in `exp_dreamerv3/`, figure `images/dreamerv3_curves.png`):
- Pendulum: solved by ~14k steps (-146 over 20 eval episodes).
- BipedalWalker-v3: solved (+303) at ~120k steps, ~10x fewer than SAC.
- BipedalWalkerHardcore-v3: +205 (20 eval episodes) at 1M steps, still rising -
  imagination helps credit assignment but does not solve rare-event exploration.
- Humanoid-v5: eval ~9000 by 1.5M steps; SAC-FORK on the same env plateaus at
  ~480 over 6M steps (entropy collapse at 17-DoF + FORK threshold tuned for
  BipedalWalker; `images/humanoid_dreamerv3_vs_sacfork.png`).
- MountainCarContinuous: fails exactly like SAC and TD-MPC2 (success 0 at 130k).
- **FetchReach-v4 (goal flattened, no HER): the key observation.** Stuck at the
  -50 floor with success 0 for ~35k steps - the reward head, trained only on
  reward it has seen, predicts -1 everywhere, so imagination gives the actor no
  gradient and the critic's imagined return just slides toward the discounted
  -1-forever asymptote (watched live: -113 -> -224 -> -279 -> -324). Once a few
  random successes entered replay it broke out and converged to -3.8 / 100 %
  within ~15k steps. TD-MPC2 solved the same setup from 5k steps.

**Why this matters for Idea 1: "model-based" is not one category.** The three
backbones now span three mechanisms on the same tasks:
DDPG+HER *manufactures* reward signal by relabelling; TD-MPC2's MPPI *searches*
at act time for reward the model knows about; DreamerV3's imagination only
*amplifies* reward already present in replay. Planning-based and
imagination-based world-model agents sit on opposite sides of the coverage
boundary. The coverage predictor generalises: early-policy achieved-goal
coverage / success density should predict DreamerV3's stall duration,
TD-MPC2's immunity, and HER's benefit with a single mechanism. The reward
head's predicted imagined return over training is a direct, cheap diagnostic
of "does the learning signal exist in the data yet" (log it per eval).

New cells this opens (roughly increasing risk):
1. **Colored collection noise in DreamerV3** (~20 lines in
   `dreamerv3/agent.py::act`: colour the actor's sampling noise across env
   steps). Does pink noise shorten/eliminate the FetchReach stall and fix
   MountainCar - i.e. does correlated noise help *because it feeds the reward
   model*, a different mechanism than helping a replay critic? No published
   work combines colored exploration noise with Dreamer-style agents
   (checked 2026-08-28; re-check before submitting). Completes the
   3-backbone x colour x HER grid.
2. **HER x DreamerV3**: relabel goals in replayed sequences (reward is
   computable from achieved/desired goal) so the reward head learns from
   hindsight successes; compare against the stall baseline. Cheap and
   directly measurable via stall length.
   *Literature check (2026-08-28): the model-based-hindsight idea is taken in
   general form - MHER (arXiv 2107.00306) relabels with goals from virtual
   model rollouts, Imaginary HER (arXiv 2110.02414) combines model-based
   imagination + curiosity + HER; both on DDPG-style backbones with separate
   dynamics models, on Fetch. GCHR (2508.06108) is the recent hindsight-
   regularisation line. Novelty therefore narrows to: (a) the RSSM/Dreamer
   instantiation, where reward comes from a learned head rather than being
   computed, and (b) the reward-head bootstrap-gap mechanism we measured -
   position any paper as explaining WHEN model-based hindsight is needed,
   not as inventing it. Also found: DreamerV3-XP (arXiv 2510.21418),
   uncertainty-driven exploration in DreamerV3 - cite as the smart-exploration
   reference point for cell 1.*
3. **Hindsight in imagination**: relabel goals inside imagined rollouts with
   an analytic goal-distance reward instead of the learned head. Higher
   novelty/risk; check the imagined-goal literature (Imagined Goals /
   PlaNet-lineage) first - and note MHER already relabels *with* imagined
   states on real transitions; the inverse (analytic reward inside the
   imagination that trains the actor) appears open.
**Cell 1 experimental design (drafted 2026-08-28).**
- *Intervention*: in `dreamerv3/agent.py::act`, replace the i.i.d. eps in
  u = mu + std * eps with a per-episode colored-noise sequence (reuse
  `sac/noise.py`), normalised to unit marginal variance per dim so only the
  temporal correlation changes, reset in `reset_episode()`. Everything else
  (world-model training, imagination, eval mean action, seed phase's uniform
  random actions) untouched. Dreamer is off-policy and the model conditions
  on executed actions, so no correction terms - a pure behaviour-policy
  ablation. beta=0 recovers the current agent exactly.
- *Pre-registered hypotheses*:
  H1 beta>0 shortens time-to-first-success on coverage-limited tasks
  (MountainCar, GoalMountainCar, Push) and does not hurt dense controls
  (Pendulum, BipedalWalker).
  H2 (the Dreamer-specific mechanism claim): the benefit is mediated by the
  reward model - stall length tracks time-to-first-success, and post-stall
  convergence rate is colour-independent. Distinguishes "pink feeds the
  reward head" from the generic "pink helps SAC" result.
  H3: the untrained-policy coverage probe predicts the per-task gain
  (same predictor as Idea 1, now spanning backbones).
- *Metrics*: time-to-first-success in replay (survival analysis - runs that
  never succeed are censored, the right stats for MountainCar); stall length
  via the reward-head diagnostic (log per eval: max decoded reward-head
  output over eval states + mean imagined return - we watched this slide
  -113 -> -324 on FetchReach); eval success/return; world-model health
  (recon, KL) to catch the confound that correlated actions reduce data
  diversity and hurt the model.
- *Grid*: beta in {0, 0.5, 1, 2} x {MountainCarContinuous, FetchReach} x
  10 seeds first (80 runs, 1-2 h each, 2-3 concurrent on the one GPU
  ~ under two weeks); then Push + GoalMountainCar + dense controls at 5
  seeds. FetchReach's stall length is a continuous outcome - much better
  statistical power than binary solve rates.
- *Completing the figure*: TD-MPC2 with colored *executed* noise (the
  teed-up ~20-line change) gives the 3-backbone x colour comparison;
  DreamerV3-XP as the uncertainty-exploration reference.
- *Pitfalls*: keep effective noise magnitude matched across beta (unit
  variance handles it, but min_std=0.1 floors the scale - report it);
  episode length T matters (the MountainCar T=200 vs T=1000 flip already
  measured); keep the uniform-random seed phase identical across cells.

**Pilot measured (2026-08-28/29, FetchReach, beta {0,1} x 3 seeds, 50k steps,
eval every 2.5k; breakout = first eval success >= 0.6 sustained).**
white: 37.5k / 45k / censored; pink: 35k / 37.5k / 50k (borderline).
Distributions overlap completely - **null on FetchReach, as H1/H3 predict**:
the repo's HER study already showed FetchReach is not coverage-limited
(random policy touches the goal ~18 % of episodes), so colour has nothing to
buy; the stall is success-*density* in replay, which correlation does not
change. The informative colour test is the coverage-limited tasks -
MountainCar beta {0,1} x 3 seeds launched next. Harness notes: the
reward_max stall diagnostic works as a leading indicator (reads ~0 when a
success is in the sampled batch, well before eval moves) but is single-batch
noisy - switch to a running max between evals before the full grid.
**MountainCar pilot + the amplitude-collapse finding (2026-08-29).**
White beta=0 x 3 seeds: flat zero for all 130k steps (matches the earlier
run). Pink beta=1 x 3 seeds: ALSO zero goal touches in training - yet the
untrained coverage probe (H3's probe, 30 episodes each) shows pink reaching
the flag 8/30 with best position +0.47 vs white 0/30 / best +0.09. The
resolution, verified by loading trained checkpoints: by the first evals the
actor has collapsed to |mu| ~ 0.00 with std pinned at the 0.1 floor for BOTH
colours - the action-penalty gradient through the reward head crushes the
exploration amplitude within the first few thousand updates, closing the
window before pink's ~4-episode expected hitting time. **Correlation only
matters if amplitude survives.** This also explains the SAC-CN vs Dreamer
discrepancy: SAC's entropy target holds sigma up; Dreamer's 3e-4 entropy
bonus cannot. The experimental surface is 2D: correlation beta x collection
amplitude. Follow-up 2x2 launched (beta {0,1} x collect_min_std 0.5, 2
seeds, collection-only std floor - training/eval untouched): prediction is
pink+floor solves, white+floor fails (white coverage is 0/30 even untrained
at wide std), either alone fails - a clean interaction effect and the
candidate headline figure.

**Third bottleneck found: terminal-transition undersampling (2026-08-29).**
Probing the collapsed actor with the floor showed pink+floor0.5 touches the
goal 6/20 episodes (floor1.0: 15/20; red beta=2 floor1.0: 9/20) - so the
running pink+floor lanes DID explore successfully, yet the reward head
stayed at the floor. Cause: MountainCar's +100 sits on the episode's FINAL
transition, and uniform-over-starts window sampling in the within-episode
SeqReplayBuffer gives a tail row ~H=32x fewer valid windows than an interior
row - terminal-only rewards are undersampled ~32x (~0.5 % of batches by
130k). Official DreamerV3 avoids this by letting sequences cross episode
boundaries. This cleanly separates FetchReach (non-terminal successes, no
bias, breakout worked) from MountainCar (terminal-only reward, starved).
Fix implemented in `dreamerv3/buffer.py::sample`: end_frac=0.25 of each
batch drawn from episode-end-aligned windows (unit-tested: terminal reward
in 97 % of batches vs ~4.5 % before). Fixed-buffer 2x2 rerun queued (mcfx_
tags, pink-first). The MountainCar causal chain is now three verified
bottlenecks: (1) amplitude collapse blocks exploration; (2) floor+pink fixes
exploration; (3) terminal undersampling starves the reward head anyway.
Each was found by a cheap targeted probe - the probes themselves are the
paper's methodological through-line.

**Final 2x2 result, fixed buffer (2026-08-29, mcfx_ runs, 130k steps).**
- pink+floor s0: first eval > 90 at **10k steps**, best +96.2, ends +92.5 -
  SOLVED. s1: reaches +97.2 (first > 90 at 20k) but oscillates between
  solved and full-throttle-no-goal (-98) - the actor overshoots the
  action-cost/goal tradeoff; reward head stays at ~+100 throughout, so this
  is post-breakout REINFORCE instability, not exploration.
- white+floor s0/s1: flat 0.0 for all 130k steps, both seeds.
Combined table across all MountainCar cells (130k budget, DreamerV3):
  plain / pink-only / floor-only(white+floor) / old-buffer pink+floor: all 0
  fixed-buffer pink+floor: solves in 10-20k steps.
**Correlation, amplitude, and terminal-reward sampling are each necessary
and only jointly sufficient.** First DreamerV3 configuration in this repo to
solve MountainCarContinuous. Follow-ups for the full grid: (a) stabilise the
actor post-breakout (entropy schedule or lower actor lr - s1's oscillation);
(b) ablate end_frac on FetchReach (near-end successes are undersampled there
too - the stall should shorten); (c) 10 seeds, plus beta 0.5/2 columns; (d)
report the probe -> outcome ladder as the method (each bottleneck was found
by a <5 min targeted probe).

Implementation (uncommitted as of 2026-08-29): noise_beta/noise_seq_len +
collect_min_std kwargs, --noise-beta / --collect-min-std flags, buffer
end_frac terminal-aligned sampling (colored eps via `sac/noise.py::ColoredNoiseProcess`,
lag-1 action autocorr 0.54 pink vs 0.14 white verified).

Practicals: this implementation turns Fetch-scale runs around in <2 h and
Humanoid overnight on one GPU, so a 5-10 seed grid is feasible. Secondary
seeds (a line, not a paper): SAC entropy collapse at 17-DoF with alpha pinned
(implementation-study material); the Hardcore rare-event result as
corroborating evidence for the mechanism taxonomy.

### Dreamer+search hybrid (measured 2026-08-30)
Added act-time MPPI through the RSSM prior to DreamerV3 (`--plan`:
TD-MPC2-style search - 512 samples + 24 actor-seeded rollouts, horizon 5,
reward-head + continue-head scoring, critic tail value, warm start; training
unchanged; ~28 ms/act). Motivation: the search-vs-amplify mechanism split -
search should consume modeled reward immediately, skipping the slow
policy-amplification phase of the FetchReach stall.

FetchReach, plan vs plain, 3 seeds, identical current code (end_frac buffer):
breakouts plan {20k, 32.5k, 35k} vs plain {30k, 42.5k, 45k} - median 32.5k
vs 42.5k, search earlier on 2/3 seeds (both by ~10-22k), later on one (5k).
Final returns a wash ({-1.8,-1.8,-9.8} vs {-2.6,-2.8,-9.2}); no censored
runs in either arm. Directionally supports the mechanism but n=3 and one
reversal - needs the 10-seed grid before claiming. Side result: the plain
arm doubles as the end_frac-on-FetchReach ablation - old-buffer white
breakouts were {37.5k, 45k, censored}, new-buffer {30k, 42.5k, 45k}:
modest improvement, right direction, within noise at n=3.

Measurement lesson: the monotone reward_max diagnostic saturates to ~0
immediately on FetchReach (seed episodes already contain ~18 % successes),
so the two-phase stall decomposition wasn't measurable as designed; a
sharper phase metric would be the reward head's prediction at goal-adjacent
states or the critic's value at episode-start states, logged per eval.

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
