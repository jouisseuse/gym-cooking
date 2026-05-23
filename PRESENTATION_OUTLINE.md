# Presentation README — outline (draft)

## Motivation — have you ever cooked with a free-rider?

- Hook: have you ever played a cooperative game with others? Mario
  Kart, Among Us, even a group project.
- Teammates have different personalities. In a group project there
  are free-riders and there are contributors. You don't get to
  observe their motivation; you only see what they do.
- But how you split work depends on what they're *going to do*. If
  I know my teammate isn't going to do task A (and A is necessary),
  I should just do it myself instead of wasting time waiting.
- So we need to **infer personality / motivation from observed
  actions**. That's the project.

## Why inference matters

- Two GIFs side-by-side, same scenario, same partner. Only difference:
  whether agent B does the inference step.
  - **Without inference**: agent B keeps assuming the partner will do
    its share; both wait around; recipe fails.
  - **With inference**: agent B realises the partner is selfish, takes
    over, recipe delivered.

*Placeholder for the two GIFs Yixi is preparing.*

This is the difference between Part 1 (agent gets ground-truth SVO for
free) and Part 2 (agent has to figure it out).

## What is SVO?

- **Social Value Orientation** — a continuous personality trait from
  social psychology measuring how much you weight your own welfare vs.
  the team's welfare.
- The classic visualisation is a **circle / angle picture**: x-axis is
  payoff to yourself, y-axis is payoff to the other person, and your
  SVO is the *angle* you'd pick on that plane.
  - θ = 0°    — pure selfish (axis pointing at "self")
  - θ = 45°   — prosocial / balanced
  - θ = 90°   — pure altruistic (axis pointing at "other")
- One number, continuous, covers the full spectrum. Has a clean
  utility form:
  ```
  U(self, other) = cos(θ) * self_payoff + sin(θ) * other_payoff
  ```
- *That single number θ is the hidden variable we want to infer.*

## The cooking game

- Multi-agent Overcooked-style gridworld. Two cooks share a kitchen,
  need to deliver a recipe (we use a tomato + lettuce salad).
- Each cook makes its own decisions, no shared memory, no chat.
- We build on the existing **"Too Many Cooks" project (Wu, Wang et
  al.)** — NeurIPS 2020 CoopAI Workshop Best Paper, CogSci 2020 Best
  Computational Modeling. GitHub: `rosewang2008/gym-cooking`.
- Their contribution: **Bayesian Delegation**. Each agent maintains a
  posterior over *who is doing which sub-task* and updates it by
  inverse planning on observed actions.
- The assumption we're relaxing: in their setup, everyone is
  identical and fully cooperative. That's the gap we plug with SVO.

## Our idea

Pull the headline content from
[`SVO_PROJECT.md`](SVO_PROJECT.md); below is a suggested two-subsection
split.

### Design

- Two layers, decoupled:
  1. **Each cook has its own SVO** → drives that cook's own decisions.
     Selfish cook actually idles. Altruistic cook actually runs to
     whatever's needed next.
  2. **Each cook infers its partner's SVO from observed actions** →
     drives delegation. If I think you're selfish, I take over your
     subtasks instead of waiting for you.
- Two halves of the project map directly to those layers:
  - **Part 1** — SVO drives behavior (the visible "selfish stays put,
    altruistic sprints" demo).
  - **Part 2** — Particle filter over partner's SVO. Closes the loop:
    posterior mean writes back into the delegator's prior.

### Implementation details

Three things worth showing in the writeup; the full spec is in
`SVO_PROJECT.md`.

1. **SVO in the planner** — utility is
   `cos(θ) * r_self + sin(θ) * r_team`. The original BD cost is
   recovered at θ = 45°.
2. **SVO in the delegator** — per-partner tilt on the spatial prior:
   allocations that put a *selfish* partner on a cooperative sub-task
   get down-weighted; allocations that put them on `None` get up-weighted.
   This is what closes the loop with Part 2's inferred θ.
3. **Particle filter (Part 2)** — mixture likelihood
   ```
   P(a | θ) = |cos θ| · P(a | None policy) + |sin θ| · P(a | BRTDP)
   ```
   The discriminator isn't BRTDP's argmin (which is "walk toward goal"
   under every θ); it's *whether the agent moves at all*. Idleness →
   selfish; purposeful movement → altruistic. If we have room for a
   single equation in the writeup, this is the one.

## Results

Three runs, same ego (altruistic), three different partners. All three
deliver. The step count depends on the partner's SVO.

| Partner | Steps to deliver | Inference (agent-1 → agent-2) |
| --- | :---: | :---: |
| selfish (θ = 0°)    | 45 | posterior settles at +7° |
| prosocial (θ = 45°) | 52 | posterior overshoots to +87° |
| altruistic (θ = 90°)| **41** | both directions converge to +85° |

Two observations to land:
- **Altruistic team is fastest** (41 < 45 < 52). More cooperation,
  more throughput.
- **Mid-range is the hardest coordination regime.** Two agents who
  half-want to do the same thing thrash. This goes *against* the
  naive "more cooperation = always better" intuition.

Inference figure to embed:

![inference traj — altruistic partner](images_svo/inference_traj_svo90.png)

What to say about it:
- Both agents start with a uniform prior over partner's SVO.
- After ~10 timesteps, both posteriors have climbed to ~80° (truth
  90°). Each agent has correctly figured out "my partner is altruistic."
- Standard deviation shrinks, ESS drops at resampling events — the
  filter is genuinely concentrating, not just lucky.
- For the selfish case the recovery is even tighter (posterior at
  +7° for truth 0°) because *staying still is a very loud signal*.

## Interactive piece — "you be the particle filter"

(Best place to fit this is at the end of Results, before the
limitations / wrap-up. By then the audience has seen what each SVO
looks like, so the guessing is meaningful instead of being a riddle.)

- Show one or two unlabelled GIFs of agents cooking together.
- Ask the audience to guess: *"is the magenta cook selfish, prosocial,
  or altruistic?"*
- Reveal the truth + the trajectory the particle filter took on the
  same scenario.
- Punchline: *you just did inverse planning in your head — that's
  what the PF is doing, except it can hold ambiguity quantitatively
  instead of switching back and forth.*

## What's still hard / future work

- **Mid-range SVO** (around 45°). Per-partner tilt isn't decisive,
  MAP allocation oscillates, recipe takes longer.
- **Bigger recipes / more agents.** Two cooks + salad worked; scaling
  to 4 cooks with longer recipes would stress the particle filter
  much harder.
- **Where Part 2 picks up**: the particle filter is stubbed in
  `gym_cooking/delegation_planner/svo_particle_filter.py` with a
  detailed TODO list and a working reference implementation in commit
  `89017fb`. Full handoff in `SVO_PROJECT.md`.

## One-line takeaway (closing)

People aren't all equally cooperative; assuming they are wastes
effort. We model that with SVO and let agents both *act on* and
*infer* each other's SVO. In the cooking environment, this lets an
altruistic agent rescue a recipe from a selfish teammate by realising
early that "they aren't going to help" and just doing the work
itself.

---

## Material checklist (for the team)

- [ ] Two opening GIFs — "fail without inference" + "succeed with inference"
- [x] Behavior GIFs (svo 0 / 45 / 90 partner) — `images_svo/svo*_partner.gif`
- [x] Inference trajectory figure — `images_svo/inference_traj_svo90.png`
- [ ] SVO angle diagram (PNG/SVG of the classic circle picture)
- [ ] (Optional) overlay plot of all three inference trajectories
- [ ] (Optional) extra unlabelled GIFs for the audience-guessing piece
