"""Sequential Monte Carlo (particle filter) over a partner's continuous SVO.

This is *Part 2* of the project: given a stream of observed partner actions
and an assumed (MAP) task allocation, maintain a posterior over the partner's
Social Value Orientation theta_j in [-pi/2, pi/2].

Model
-----
    Hidden:        theta_j ~ Uniform(-pi/2, pi/2)    (default prior)
    Likelihood:    P(a_{j,t} | s_t, theta_j) = softmax_beta( -Q_j^{theta_j}(s_t, .) )[a_{j,t}]

where Q_j^{theta_j} comes from a fresh copy of *this* agent's BRTDP planner
configured with the partner's hypothesized subtask and SVO theta_j (Level-1
inverse planning).

SMC update (Bayesian filter)
----------------------------
For each timestep t after observing the partner's action a_{j,t}:
    for k in 1..N:
        Q_k = Q_j^{theta_k}(s_{t-1}, .)
        ell_k = softmax_beta( -Q_k )[a_{j,t}]
        w_k <- w_k * ell_k
    w <- w / sum(w)
    if ESS(w) < ess_threshold:
        resample (multinomial) and add Gaussian jitter to particles

See ``SVO_PROJECT.md`` for the full specification and design rationale.
"""
import copy
import numpy as np
import scipy as sp

from navigation_planner.utils import get_single_actions


class SVOParticleFilter:

    @staticmethod
    def none_action_prob_for(theta):
        """Same SVO-dependent none_action_prob as RealAgent uses. Strongly
        selfish (theta ~ 0) agents on the None subtask stay put nearly
        always; neutral/altruistic agents wiggle. Keeping the two formulas
        in sync is essential for the mixture-likelihood to reflect the
        partner's actual policy."""
        return max(0.5, abs(np.cos(theta)) ** 0.5)

    def __init__(self, partner_name, planner_template, n_particles=64,
                 beta=1.3, prior_range=(0.0, np.pi / 2),
                 jitter_sd=0.05, ess_threshold_frac=0.5, rng=None):
        # NOTE: Prior is non-negative [0, pi/2] because the delegator tilt,
        # the none_action_prob, and the mixture-likelihood weights are all
        # symmetric in theta via abs(cos) / abs(sin). So +theta and -theta
        # produce identical observable behavior, and the PF cannot
        # discriminate the sign. Constraining the prior to [0, pi/2] keeps
        # all the probability mass in the half-axis we can identify.
        """
        Args:
            partner_name: Str name of the partner whose SVO we infer.
            planner_template: An ``E2E_BRTDP`` instance to copy from. Each
                particle gets a fresh copy parameterized by its theta.
            n_particles: Number of SMC particles N.
            beta: Boltzmann rationality used in the likelihood.
            prior_range: (low, high) interval for the uniform prior on theta.
            jitter_sd: Std of Gaussian noise added after resampling.
            ess_threshold_frac: Resample when ESS < this fraction of N.
            rng: Optional ``np.random.Generator`` for reproducibility.
        """
        self.partner_name = partner_name
        self.planner_template = planner_template
        self.n_particles = int(n_particles)
        self.beta = float(beta)
        self.prior_low, self.prior_high = prior_range
        self.jitter_sd = float(jitter_sd)
        self.ess_threshold = float(ess_threshold_frac) * self.n_particles
        self.rng = rng if rng is not None else np.random.default_rng()

        # Draw particles from the prior; uniform weights.
        self.particles = self.rng.uniform(
                self.prior_low, self.prior_high, size=self.n_particles)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

        # Posterior trace for diagnostics.
        self.trace = []  # list of (mean, std, ess) tuples per update

    # ------------------------------------------------------------------ API

    def update(self, obs_tm1, action_tm1, partner_subtask,
               partner_subtask_agents):
        """One SMC update from a single observed partner action.

        If ``partner_subtask`` is None we skip the update (the None policy is
        SVO-agnostic, so the observation carries no information about theta).
        """
        if partner_subtask is None:
            self._record_trace()
            return

        log_w = np.log(self.weights + 1e-300)

        for k in range(self.n_particles):
            ll_k = self._likelihood(
                    obs_tm1=obs_tm1,
                    action_tm1=action_tm1,
                    partner_subtask=partner_subtask,
                    partner_subtask_agents=partner_subtask_agents,
                    theta=float(self.particles[k]))
            log_w[k] += np.log(ll_k + 1e-300)

        # Normalize in log space.
        log_w -= np.max(log_w)
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()

        # Resample if effective sample size has degenerated.
        if self.ess() < self.ess_threshold:
            self._resample()

        self._record_trace()

    def posterior_mean(self):
        return float(np.sum(self.weights * self.particles))

    def posterior_std(self):
        m = self.posterior_mean()
        return float(np.sqrt(np.sum(self.weights * (self.particles - m) ** 2)))

    def map_estimate(self):
        return float(self.particles[int(np.argmax(self.weights))])

    def ess(self):
        return float(1.0 / np.sum(self.weights ** 2))

    # ----------------------------------------------------------- internals

    def _likelihood(self, obs_tm1, action_tm1, partner_subtask,
                    partner_subtask_agents, theta):
        """Per-particle likelihood P(a | s, theta) via a *mixture* of:

          * the None-subtask random policy (used by an idle agent), with
            weight |cos(theta)|,
          * the BRTDP-Boltzmann policy on the observer's MAP cooperative
            subtask, with weight |sin(theta)|.

        The mixture reflects the fact that **the partner's actual choice of
        action is gated by their delegator's subtask routing**, which depends
        on theta: a selfish (low theta) agent picks ``None`` (idle) most of
        the time; an altruistic (high theta) agent picks cooperative.
        A pure BRTDP likelihood -- the previous implementation -- can't
        discriminate theta values because BRTDP's *optimal action* is
        essentially the same across theta ("walk toward goal"), so it
        misses the dominant evidence about partner SVO.

        Joint-subtask handling: if the MAP allocation puts the partner on a
        joint subtask, we configure the BRTDP copy as if the partner were
        solo on it (so action signatures match).
        """
        c = abs(float(np.cos(theta)))
        s = abs(float(np.sin(theta)))
        total = c + s + 1e-10
        w_none = c / total
        w_coop = s / total

        # ---- P(a | None policy) --------------------------------------------
        # The agent's None branch picks (0,0) w.p. NONE_ACTION_PROB and
        # spreads the rest uniformly over the other valid single-agent
        # actions.
        sim_partner = next(
                a for a in obs_tm1.sim_agents if a.name == self.partner_name)
        valid = get_single_actions(env=obs_tm1, agent=sim_partner)
        if action_tm1 == (0, 0):
            p_none = self.none_action_prob_for(theta)
        elif action_tm1 in valid:
            denom = max(1, len(valid) - 1)
            p_none = (1.0 - self.none_action_prob_for(theta)) / denom
        else:
            p_none = 1e-6

        # ---- P(a | BRTDP under theta on a cooperative subtask) ------------
        if partner_subtask is None:
            # Observer's MAP also said None -- defer to the None policy.
            return float(w_none * p_none + w_coop * p_none)

        planner = copy.copy(self.planner_template)
        planner.set_svo(theta)
        # Use the SAME cost the real partner planner uses (original BD cost).
        # Discrimination comes purely through the mixing weights.
        planner.use_svo_cost = False
        solo_agents = (self.partner_name,)
        planner.set_settings(
                env=copy.copy(obs_tm1),
                subtask=partner_subtask,
                subtask_agent_names=solo_agents)
        state = planner.start
        actions = planner.get_actions(state_repr=state.get_repr())
        if action_tm1 not in actions:
            # Distance-based fallback: at low BRTDP caps the planner may
            # not have considered every joint configuration. Score based
            # on whether the observed action moves closer to the subtask
            # target rather than returning a hard 1e-6, which otherwise
            # systematically biases the posterior away from high-theta
            # particles (where w_coop ~= 1 makes them rely on p_coop).
            p_coop = self._distance_based_p_coop(
                    planner=planner, state=state, action_tm1=action_tm1)
        else:
            q_vals = np.array([
                planner.Q(state=state, action=a, value_f=planner.v_l)
                for a in actions])
            probs = sp.special.softmax(-self.beta * q_vals)
            p_coop_brtdp = float(probs[actions.index(action_tm1)])
            # Blend BRTDP softmax with distance-based "purposeful" score.
            # If BRTDP is wishy-washy (low cap, sparse signal), distance
            # gives a robust progress proxy that still discriminates.
            p_coop_dist = self._distance_based_p_coop(
                    planner=planner, state=state, action_tm1=action_tm1)
            p_coop = 0.5 * p_coop_brtdp + 0.5 * p_coop_dist

        return float(w_none * p_none + w_coop * p_coop)

    @staticmethod
    def _distance_based_p_coop(planner, state, action_tm1):
        """Distance-shaping proxy for P(a | cooperative subtask under theta).

        Returns a higher value when ``action_tm1`` reduces BFS distance to
        the partner's subtask goal. This is the same shaping signal the
        SVO BRTDP cost uses, applied directly at the action level so it
        works even when BRTDP itself has not converged.
        """
        try:
            d_before = planner._subtask_distance(state)
            next_state = planner.T(state_repr=state.get_repr(), action=action_tm1)
            d_after = planner._subtask_distance(next_state)
        except Exception:
            return 0.25  # mildly informative fallback
        progress = float(d_before - d_after)
        # Sigmoid-shaped scoring: closer-to-goal -> p_coop ~ 1, away -> ~0.
        # Centred at 0 so neutral actions (no distance change) -> 0.5.
        return float(1.0 / (1.0 + np.exp(-2.0 * progress)))

    def _resample(self):
        idx = self.rng.choice(
                self.n_particles, size=self.n_particles,
                replace=True, p=self.weights)
        new_particles = self.particles[idx]
        # Add small Gaussian jitter so the support keeps exploring.
        new_particles = new_particles + self.rng.normal(
                0.0, self.jitter_sd, size=self.n_particles)
        new_particles = np.clip(new_particles, self.prior_low, self.prior_high)
        self.particles = new_particles
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    def _record_trace(self):
        self.trace.append((self.posterior_mean(),
                           self.posterior_std(),
                           self.ess()))
