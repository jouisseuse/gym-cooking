"""Sequential Monte Carlo (particle filter) over a partner's continuous SVO.

================================================================
    PART 2 -- TO BE IMPLEMENTED BY PARTNER
================================================================

Part 1 of the project (decision-making with a *known* partner SVO) is fully
in place. Each agent's planner and delegator already react to its own SVO
and to its current estimate of every partner's SVO. In Part 1 those
partner estimates come from the CLI ground truth (``--svo1`` etc.), so the
inferring agent gets the partners "for free."

Part 2 replaces that oracle with a particle filter that *infers* each
partner's SVO from observed actions. When you implement the methods below
and the ``--infer-svo`` flag is set, the rest of the codebase will start
using the filter automatically -- the delegator already calls
``pf.update(...)`` each step and slow-blends ``pf.posterior_mean()`` into
``partner_svo_estimates``, closing the inference loop into Level-1
planning.

A working reference implementation lived on this file before the strip.
If you get stuck, run:
    git show 89017fb:gym_cooking/delegation_planner/svo_particle_filter.py

For the full math, see ``SVO_PROJECT.md`` Section 4. Quick recap below.

----------------------------------------------------------------
Model
----------------------------------------------------------------
Hidden variable:
    theta_j in [0, pi/2]                    (partner j's SVO magnitude)

    Note: the prior is non-negative because the rest of the model
    (delegator tilt, none_action_prob, mixture weights) is symmetric in
    theta via abs(cos) / abs(sin). So +theta and -theta produce identical
    observable behavior and the PF cannot identify the sign. The default
    ``prior_range=(0, pi/2)`` keeps probability mass in the identifiable
    half-axis.

Prior (default):
    theta_j ~ Uniform(0, pi/2)

Likelihood -- *mixture* of partner's two possible policies, weighted by
theta. The partner's actual action is gated by their delegator's
SVO-tilted subtask routing: a low-theta agent mostly picks ``None``
(random None-policy); a high-theta agent picks cooperative (BRTDP).

    P(a_{j,t} | s_t, theta_j) =
          |cos(theta_j)|  *  P(a | None policy under theta_j)
        + |sin(theta_j)|  *  P(a | BRTDP-Boltzmann on partner_subtask)

A pure BRTDP-only likelihood (without the None branch) cannot
discriminate theta values because BRTDP's argmin is "walk toward goal"
under almost every theta. The mixture is what creates a usable signal:
idle observations strongly favor low theta; purposeful moves strongly
favor high theta.

The None-policy probability also depends on theta (see
``RealAgent.__init__`` in ``utils/agent.py``: low-theta agents have
``none_action_prob`` close to 1 so they truly idle). The static method
``SVOParticleFilter.none_action_prob_for(theta)`` returns the same value
so the mixture matches the real policy. Keep the two formulas in sync.

----------------------------------------------------------------
SMC algorithm
----------------------------------------------------------------
Init:
    particles theta^{(k)} ~ Uniform(prior_range)
    weights   w^{(k)}     = 1/N

For each timestep t after observing partner action a_{j,t}:
    1. For each particle k:
            ell_k = self._likelihood(obs_tm1, a_{j,t}, partner_subtask,
                                     partner_subtask_agents, theta=theta^{(k)})
            w_k  <- w_k * ell_k
    2. Normalise w (use log-space for stability).
    3. If ESS = 1 / sum(w_k^2) < ess_threshold_frac * N:
           multinomial resample, add Gaussian jitter ~ N(0, jitter_sd^2),
           clip to prior range, reset weights to uniform.
    4. Append (mean, std, ess) to self.trace for diagnostics.

----------------------------------------------------------------
How this integrates (already wired)
----------------------------------------------------------------
1. ``BayesianDelegator.bayes_update`` calls ``_update_svo_filters`` when
   ``--infer-svo`` is on. That helper lazily creates one
   ``SVOParticleFilter`` per partner, then calls
   ``pf.update(obs_tm1, action_tm1, subtask, subtask_agent_names)`` each
   step. It also passes a fallback cooperative subtask when the
   SVO-tilted MAP routes the partner to None.
2. After ``update``, the delegator slow-blends
   ``pf.posterior_mean()`` into ``self.partner_svo_estimates[partner_name]``
   (alpha=0.15). ``get_other_agent_planners`` reads that dict and
   reconfigures each partner-planner copy with the blended SVO.

You shouldn't need to edit anything outside this file. Just implement the
methods marked ``TODO (part2)`` below.

----------------------------------------------------------------
Recommended test recipe
----------------------------------------------------------------
* Two agents, partner SVO = 0 deg, ego SVO = 90 deg, --infer-svo on.
  After ~20 timesteps the posterior should settle near 0 deg with small
  std (selfish agent is mostly idle, strong signal).
* Partner SVO = 90 deg should show the posterior climbing toward +90
  over ~30 timesteps (slower than the selfish case because purposeful
  movement is a weaker discriminator than idleness).
* If the posterior never moves: check that ``_likelihood`` is calling
  ``planner.set_svo(theta)`` before ``planner.set_settings`` and that
  ``planner.use_svo_cost`` is False (we score against the partner's
  real cost; SVO only enters through the mixing weights).
"""
import copy
import numpy as np
import scipy as sp

from navigation_planner.utils import get_single_actions


class SVOParticleFilter:

    @staticmethod
    def none_action_prob_for(theta):
        """Match ``RealAgent.__init__``'s SVO-dependent ``none_action_prob``
        so the likelihood mixture reflects the partner's real None-policy:
        selfish agents stay put nearly always; neutral/altruistic agents
        would wiggle if they ever ended up on None.

        Keep this in sync with ``utils/agent.py`` -- if you change one,
        change both.
        """
        return max(0.5, abs(np.cos(theta)) ** 0.5)

    def __init__(self, partner_name, planner_template, n_particles=64,
                 beta=1.3, prior_range=(0.0, np.pi / 2),
                 jitter_sd=0.05, ess_threshold_frac=0.5, rng=None):
        """See module docstring for the SMC spec.

        Args:
            partner_name: Str name of the partner whose SVO we infer.
            planner_template: An ``E2E_BRTDP`` instance to copy from in
                ``_likelihood``. Each particle needs a fresh copy.
            n_particles: Number of SMC particles N.
            beta: Boltzmann rationality used in the likelihood.
            prior_range: (low, high) interval for the uniform prior. The
                default is non-negative because the model is sign-symmetric
                in theta -- see module docstring.
            jitter_sd: Std of Gaussian noise added to particles after
                resampling.
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

        # TODO (part2): initialise ``self.particles`` (np.ndarray, shape (N,))
        # from the uniform prior on [self.prior_low, self.prior_high], and
        # ``self.weights`` (np.ndarray, shape (N,)) uniform 1/N.
        self.particles = None
        self.weights = None

        # Diagnostics: append (mean, std, ess) tuple after every update.
        # Used by misc/metrics/plot_svo_inference.py.
        self.trace = []

    # ------------------------------------------------------------------ API

    def update(self, obs_tm1, action_tm1, partner_subtask,
               partner_subtask_agents):
        """One SMC step from one observed partner action.

        When ``partner_subtask`` is None the cooperative branch of the
        mixture collapses; defer to the None-policy term. The delegator
        usually passes a *fallback* cooperative subtask instead, so this
        case is rare.

        TODO (part2): implement the SMC step:
            1. For each particle k, multiply w_k by
               self._likelihood(obs_tm1, action_tm1, partner_subtask,
                                partner_subtask_agents,
                                theta=self.particles[k]).
               Work in log-space.
            2. Normalise weights.
            3. If self.ess() < self.ess_threshold, call self._resample().
            4. Append (self.posterior_mean(), self.posterior_std(),
                       self.ess()) to self.trace.
        """
        raise NotImplementedError(
                "SVOParticleFilter.update is the Part 2 deliverable. "
                "See module docstring for the SMC spec.")

    def posterior_mean(self):
        """TODO (part2): weighted mean of self.particles under self.weights."""
        raise NotImplementedError

    def posterior_std(self):
        """TODO (part2): weighted std of self.particles under self.weights."""
        raise NotImplementedError

    def map_estimate(self):
        """TODO (part2): particle with maximum weight."""
        raise NotImplementedError

    def ess(self):
        """TODO (part2): effective sample size = 1 / sum(w_k^2)."""
        raise NotImplementedError

    # ----------------------------------------------------------- internals

    def _likelihood(self, obs_tm1, action_tm1, partner_subtask,
                    partner_subtask_agents, theta):
        """Per-particle mixture likelihood. See module docstring.

        TODO (part2): implement the mixture
            P(a | theta) = w_none * P(a | None) + w_coop * P(a | BRTDP)
        where:
            w_none = |cos(theta)| / (|cos(theta)| + |sin(theta)| + eps)
            w_coop = |sin(theta)| / (|cos(theta)| + |sin(theta)| + eps)

            P(a | None) under theta:
                p_stay = self.none_action_prob_for(theta)
                if a == (0, 0):  p_none = p_stay
                elif a in get_single_actions(obs_tm1, partner sim_agent):
                    p_none = (1 - p_stay) / (n_valid - 1)
                else:             p_none = 1e-6

            P(a | BRTDP under partner_subtask):
                copy self.planner_template, set_svo(theta),
                use_svo_cost=False, configure with subtask=partner_subtask
                and subtask_agent_names=(self.partner_name,).
                Compute Q values for valid actions, softmax(-beta * Q).
                If action not in valid set, fall back to a distance-based
                proxy (sigmoid of progress toward subtask goal) -- BRTDP
                may not have explored every joint configuration at low
                caps, and a hard 1e-6 fallback would systematically bias
                the posterior away from high-theta particles.
        """
        raise NotImplementedError

    def _resample(self):
        """Multinomial resampling with Gaussian jitter.

        TODO (part2):
            idx = self.rng.choice(N, size=N, replace=True, p=self.weights)
            new = self.particles[idx] + self.rng.normal(0, self.jitter_sd, N)
            self.particles = np.clip(new, self.prior_low, self.prior_high)
            self.weights = np.full(N, 1.0 / N)
        """
        raise NotImplementedError
