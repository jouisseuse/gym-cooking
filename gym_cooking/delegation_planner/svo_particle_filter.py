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


class SVOParticleFilter:

    def __init__(self, partner_name, planner_template, n_particles=64,
                 beta=1.3, prior_range=(-np.pi / 2, np.pi / 2),
                 jitter_sd=0.05, ess_threshold_frac=0.5, rng=None):
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
        """Per-particle likelihood P(a | s, theta) via Boltzmann over Q.

        Uses a fresh BRTDP copy so the value-function cache is theta-clean.

        Joint-subtask handling: if the MAP allocation puts the partner on a
        joint subtask with someone else, we configure the planner as if the
        partner were the *only* one on the subtask. This scores the partner's
        single action under their own Q values (a slight approximation that
        avoids needing the other agent's action in the joint signature).
        """
        planner = copy.copy(self.planner_template)
        planner.set_svo(theta)
        # Real planning uses the original (SVO-free) cost; the PF needs SVO
        # in cost so that Q values actually differ across particles.
        planner.use_svo_cost = True
        # Force solo planning: treat the partner as the only agent on the
        # subtask. This guarantees ``get_actions`` returns single-agent moves
        # that match the partner's observed action signature.
        solo_agents = (self.partner_name,)
        planner.set_settings(
                env=copy.copy(obs_tm1),
                subtask=partner_subtask,
                subtask_agent_names=solo_agents)

        state = planner.start
        actions = planner.get_actions(state_repr=state.get_repr())

        # If the observed action is no longer in the action set (agent vanished,
        # collision-pruned, etc.), return a small but non-zero number.
        if action_tm1 not in actions:
            return 1e-6

        q_vals = np.array([
            planner.Q(state=state, action=a, value_f=planner.v_l)
            for a in actions])
        # BRTDP minimizes cost, so the soft-rational policy is softmax over -Q.
        probs = sp.special.softmax(-self.beta * q_vals)
        return float(probs[actions.index(action_tm1)])

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
