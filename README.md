# Inferring Social Value Orientation in Multi-Agent Cooking Coordination

![headline img](images_svo/image.png)

## Contributors  
- Bufan Gao
- Tianrun Wu
- Yufei Mao
- Jiabin Zou

## Introduction

> "Time's up! Holding a hotplate, don't mind me! Just burning my hands!"
> -Overcooked Player  
> "Ruined more friendships than +4 cards in uno, monopoly, and blue shells combined."
> -Anonymous Overcooked Review  

### 1. Motivation: Ever teamed with a Free-Rider?   
Teamwork matters in real-world tasks: conversation, group projects, cooperative games, etc. However, people have different personality and teammates **do NOT all contribute in the same way**. Some people prioritize team tasks, some hesitate, while some couldn't care less. Crucially, we can't read our teammate's mind directly. We have to **infer their tendencies** from observing their behaviors, and then make our own plans.

**Wu et al. (2021)** formalizes the problem with their **Bayesian Delegation** model, using an overcooked-inspired cooking game gridworld. Their model helps artificial agents coordinate in a cooking game by watching each other's actions and guessing what task the other agent is working on. In this way, the agets can divide labor without needing to talk to each other.

![Wu_paper img](images_svo/Wu.jpg)

### 2. Current Project: heterogenous social preferences    
Our project extends on Wu et al.'s model by giving each cook a continuous **Social Value Orientation (SVO)** trait `theta` that controls how much it values its own effort versus team progress, and a partner can *infer* that trait from observed behavior using a particle filter.

> Built on top of [rosewang2008/gym-cooking](https://github.com/rosewang2008/gym-cooking)
> — *"Too many cooks: Bayesian inference for coordinating multi-agent collaboration."*
> Wu, S. A., Wang, R. E., Evans, J. A., Tenenbaum, J. B., Parkes, D. C.,
> Kleiman-Weiner, M. (2021). *Topics in Cognitive Science.*

## The core idea

Each agent's social preference is represented with `theta ∈ [0, pi/2]`;  
Intuitively, `theta` controls how much the ageny weights its own effort cost versus team progress:  
```
U_i(s, a; theta) = cos(theta) * r_self_i(s, a)  +  sin(theta) * r_team(s, a)
```

| `theta` | Interpretation | Observable Behavior |
| --- | --- | --- |
| **0°** | Selfish | Sits at the corner, doesn't pick anything up, lets the partner cook alone |
| **45°** | Prosocial (≈ original BD) | Splits sub-tasks with the partner |
| **90°** | Altruistic | Walks to the next useful object every step, joins the partner at merges |

Knowing one's own social preference doesn't solve the problems; we have to know what the others think to act in accordance. Each cook *also* maintains a belief about every partner's `theta` and uses the inferred value to decide who should do what. 

## Project Overview

Our project has two connected parts:

### Part 1: Decision-making with *known* SVO  
Each cook is assigned an SVO value. The value affects both the agent's own planning and how the delegator routes tasks to that agent.  
- The ego agent uses its belief about the partner's SVO to decide whether to wait for the partner or take over the work itself.  
- A "selfish" partner is more likely to be assigned `None`, and an altruistic partner is more likely to be assigned useful cooperative subtasks.

#### Behavior demo

`open-divider_salad` (Tomato + Lettuce → chop both → merge → plate →
deliver). Ego (agent-1, **blue cook**) is fixed at **theta = 90°** in
all three runs; the partner (agent-2, **magenta cook**) varies. All
three **deliver the recipe**; only the step count and the partner's
behavior differ.

|  Selfish partner (`theta_2 = 0°`)  |  Prosocial partner (`theta_2 = 45°`)  |  Altruistic partner (`theta_2 = 90°`)  |
| :---: | :---: | :---: |
| ![selfish](images_svo/svo0_partner.gif) | ![prosocial](images_svo/svo45_partner.gif) | ![altruistic](images_svo/svo90_partner.gif) |
| **45 steps.** Magenta stays put; blue carries the whole recipe alone. | **52 steps.** Both move; mid-range SVO has the most coordination friction. | **41 steps.** Tight cooperation -- both chop in parallel and meet at the plate. |

### Part 2: Inferring *unknown* SVO  
The partner's SVO is treated as hidden. Each agent maintains a particle-filter posterior over the partner's `theta`. After every observed partner action, the filter updates which candidate SVO values best explain the behavior. 
- Idle actions are stronger evidence for lower `theta`; purposeful movement towards useful subtasks is stronger evidence for higher `theta`.
- The posterior mean is fed back into the delegator, closing the loop between inference and planning.  

Key observations:

- **Visible selfish ≠ visible altruistic.** The selfish partner literally
  doesn't help; the altruistic one sprints to whatever is next needed.
- **More cooperation → faster delivery at the extremes**: 41 (altruistic)
  vs 45 (selfish), a 9% speedup despite the extra coordination overhead.
- **Mid-range (theta=45) is the hardest coordination regime** — ironically
  the slowest of the three because both agents partially want to do the
  same task and the per-partner SVO tilt isn't decisive enough to split
  roles cleanly.

## Inference demo (Part 2 — TODO, reference run)

The figure below was generated from a working reference implementation
of the particle filter (commit `89017fb`, before the strip). Both
agents have `theta = 90°` and each infers the other's SVO from observed
actions:

![inference traj — altruistic partner](images_svo/inference_traj_svo90.png)

Both posteriors climb from the prior mean (~45°) toward ~85° (truth
90°) and the std shrinks as evidence accumulates. ESS drops as the
posterior concentrates, with brief jumps at resampling events.

For the selfish-partner condition the posterior settles at +7°
(truth 0°) within 45 steps; for prosocial it overshoots to +87°
(truth 45°) — prosocial actions look very similar to altruistic when
the partner is also moving.

## Model specification

### Notation

| Symbol | Meaning |
| --- | --- |
| `s_t` | World state (positions, holdings, object states). |
| `a_{i,t}` | Agent `i`'s primitive move at step `t`. |
| `theta_i in [0, pi/2]` | Agent `i`'s SVO (fixed per episode, hidden to others). |
| `beta` | Boltzmann rationality. Defaults to `arglist.beta = 1.3`. |

### Generative model (Part 1)

Per-agent utility:
```
U_i(s, a; theta_i) = cos(theta_i) * r_self_i(s, a)  +  sin(theta_i) * r_team(s, a)

r_self_i(s, a) = -(time_cost + action_cost * 1{a_i != (0,0)})
r_team(s, a)   = -lambda * d_after(s, a)
```
where `d_after` is the BFS distance from the post-action position to
the next subtask-relevant object. Real action selection uses the
original BD cost (`use_svo_cost=False`); the SVO-weighted cost is
reserved for the particle filter's per-particle likelihood evaluation.

**Delegator SVO tilt.** Each candidate allocation is weighted by, for
*every* agent `j` in the allocation:

```
tilt(j, subtask) = |sin(theta_j)|   if subtask is cooperative
                 = |cos(theta_j)|   if subtask == None
```

`theta_j = self.svo` for self and `partner_svo_estimates[j]` for
partners. An altruistic ego that believes its partner is selfish will
have a MAP allocation that puts the partner on `None` and assigns
itself every cooperative subtask — that's what makes the
selfish-partner condition deliver at all.

**SVO-dependent `none_action_prob`.** A selfish agent on the None
subtask needs to *actually* stay put, otherwise its random-walk
None-policy accidentally triggers `interact()` and the agent looks
like it's helping. We scale
`none_action_prob = max(0.5, |cos(theta)|^0.5)` so `theta=0` gives
`prob=1.0` (always `(0,0)`).

**Anti-deadlock prune.** When *any* plate in the world already has an
ingredient on it (e.g. `ChoppedTomato-Plate`), allocations of the form
`Merge(<single ingredient>, Plate)` are pruned. Without this, two
cooperative agents grab a plate each, plate one ingredient apiece,
and deadlock with two half-finished plates.

### Inference model (Part 2, **TODO**)

Hidden:   `theta_j ~ Uniform(0, pi/2)`

The prior is non-negative because the rest of the model is **symmetric
in sign** (`|sin|`, `|cos|` everywhere). A `[-pi/2, pi/2]` prior would
leak probability mass into the negative half that no observation can
identify.

Likelihood (mixture of None-policy and BRTDP-Boltzmann under the
partner's subtask):
```
P(a_{j,t} | s_t, theta_j) =
      |cos(theta_j)|  *  P(a | None policy under theta_j)
    + |sin(theta_j)|  *  P(a | BRTDP-Boltzmann on partner_subtask)
```

The mixing weights are the key discriminator. A pure-BRTDP likelihood
fails because BRTDP's argmin is "walk toward goal" under almost every
theta. Mixing in the None-policy (weighted by `|cos(theta)|`) creates
real signal: idle observations strongly favor low theta; purposeful
moves strongly favor high theta.

SMC update: see the module docstring of
[`gym_cooking/delegation_planner/svo_particle_filter.py`](gym_cooking/delegation_planner/svo_particle_filter.py).

## Repo layout

```
gym_cooking/
  main.py                                # CLI: --svoN, --infer-svo, --n-particles
  utils/
    svo.py                               # SVO presets, parser
    agent.py                             # RealAgent reads theta; SVO-dep none_action_prob
  navigation_planner/planners/
    e2e_brtdp.py                         # opt-in SVO-weighted cost; team_progress shaping
  delegation_planner/
    bayesian_delegator.py                # per-partner SVO tilt; anti-deadlock prune; PF hook
    svo_particle_filter.py               # ** Part 2 -- TODO stub **
  misc/metrics/
    metrics_bag.py                       # logs SVO state per step
    make_gif.py                          # frames -> GIF
    plot_svo_inference.py                # plots PF posterior trajectory
  experiments/run_svo.py                 # SVO sweep
images_svo/                              # figures (3 behavior GIFs + 1 inference plot)
SVO_PROJECT.md                           # this document
```

## How to run

From `gym_cooking/`:

```bash
# Behavior demo: altruistic ego, selfish partner
python main.py --num-agents 2 --level open-divider_salad \
    --model1 bd --model2 bd --svo1 90 --svo2 0 \
    --max-num-timesteps 100 --cap 25 --main-cap 15 \
    --seed 1 --record

# Once Part 2 lands, enable inference:
python main.py ... --infer-svo --n-particles 16
# (Until then it prints a one-line warning and falls back to the
#  CLI ground-truth partner_svo_estimates.)
```

Make a GIF from saved frames:
```bash
python -m misc.metrics.make_gif \
    --frames misc/game/record/<run_name> \
    --out ../images_svo/<file>.gif --duration 250
```

Plot the PF inference trajectory (needs Part 2):
```bash
python -m misc.metrics.plot_svo_inference \
    --pickle misc/metrics/pickles/<run>.pkl \
    --out ../images_svo/inference_traj.png
```

## TODO list for Part 2

Everything you need is in
[`gym_cooking/delegation_planner/svo_particle_filter.py`](gym_cooking/delegation_planner/svo_particle_filter.py).
The methods marked `TODO (part2)` are:

1. `__init__` — initialise `self.particles` from the uniform prior and
   `self.weights` uniform `1/N`.
2. `update(...)` — one SMC step (per-particle likelihood, normalise,
   resample on low ESS, append diagnostic to `self.trace`).
3. `posterior_mean / posterior_std / map_estimate / ess`.
4. `_likelihood(...)` — the mixture
   `|cos(theta)| * P(a | None) + |sin(theta)| * P(a | BRTDP under theta)`.
   `SVOParticleFilter.none_action_prob_for(theta)` is already provided
   and matches the partner's real None-policy.
5. `_resample` — multinomial with Gaussian jitter, clip to prior range,
   reset weights.

The integration is already wired: `BayesianDelegator._update_svo_filters`
calls `pf.update(...)` once per step per partner, then slow-blends
`pf.posterior_mean()` into `partner_svo_estimates[partner]`
(alpha=0.15). The bag logger snapshots `pf.particles / weights / mean /
std / ess` each step.

A working reference implementation lived on this file in commit
`89017fb` (before the strip):
```bash
git show 89017fb:gym_cooking/delegation_planner/svo_particle_filter.py
```

### Sanity checks for Part 2

- Partner SVO = 0°, ego SVO = 90°, `--infer-svo` on: posterior should
  settle near 0° with small std within ~20 timesteps.
- Partner SVO = 90°: posterior should climb toward +90° over ~25
  timesteps (this is the figure embedded above).
- If the posterior never moves: check that `_likelihood` is calling
  `planner.set_svo(theta)` before `planner.set_settings`.

## Caveats and known limitations

- **Mid-range SVO is the hardest case.** With `theta=45`, the
  per-partner tilt is `|sin|=|cos|=0.707` so MAP allocation can flip
  between roles step-to-step, leading to thrashing.
- **BRTDP caps are low for speed** (`--cap 25 --main-cap 15`).
  Production runs should use the upstream defaults
  (`--cap 75 --main-cap 100`).
- **Single seed.** All figures are from `--seed 1`. A proper sweep
  is left as a follow-up.

## Citation

```
@article{wu_wang2021too,
  author = {Wu, Sarah A. and Wang, Rose E. and Evans, James A. and
            Tenenbaum, Joshua B. and Parkes, David C. and
            Kleiman-Weiner, Max},
  title  = {Too many cooks: Coordinating multi-agent collaboration
            through inverse planning},
  journal = {Topics in Cognitive Science},
  year   = {2021},
  doi    = {10.1111/tops.12525},
}
```
