"""
active_inference_agent.py — Deep active inference (Millidge 2019) via PC networks.

Algorithm from "Deep Active Inference as Variational Policy Gradients" (Millidge, 2019).
PC networks substitute for standard backprop networks; learning rule is otherwise identical.

Four networks:
  transition_net : p(s'|s,a)           k_lut=[dim_s, h_t, dim_s+dim_a]
  efe_net        : G_ψ(s)    (online)  k_lut=[n_actions, h_e, dim_s]
  efe_target     : G_target  (frozen)  same architecture, synced every target_update_freq steps
  policy_net     : π(a|s) logits       k_lut=[n_actions, h_p, dim_s]

G convention: EFE is a COST — lower is preferred.
  G_hat = -r_t + eps_transition + discount * E_{a'~π}[G_target(s_t)[a']]
  Terminal: G_hat = -r_t + eps_transition  (G_bootstrap = 0 at episode end)

Per-timestep schedule:
  1. Learn transition on (s_{t-1},a_{t-1}) → s_t
  2. Compute G_hat TD target (frozen target network bootstraps from s_t)
  3. Online EFE update: train only the chosen action head G_ψ(s_{t-1})[a_{t-1}] → G_hat
  4. Push (s_{t-1},a_{t-1},r,s_t,done,eps_trans) to replay buffer
  5. At episode end, replay: sample batch, recompute G_hat from current target, update G_ψ
  6. Every few episodes, train policy_net on trajectory states using current EFE preferences
  7. Select action from policy_net (fallback to direct EFE early in training)
  8. Sync target every target_update_freq gradient steps

Replay prevents catastrophic forgetting: without it the online PC updates let the
output-layer bias chase the last seen G_hat, erasing state discrimination.
"""
from __future__ import annotations

import collections
import numpy as np
from typing import Optional

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pc_network import PCNet3Layer


class ActiveInferenceAgent:
    """
    Online deep active inference agent backed by PC network instances.

    Parameters
    ----------
    dim_obs         : observation dimensionality (= dim_s for fully-observed MDPs)
    dim_s           : latent state dimensionality
    dim_a           : action dimensionality
    hidden_trans    : hidden layer width for transition net
    hidden_efe      : hidden layer width for EFE net
    hidden_policy   : hidden layer width for policy net
    gamma_pc        : PC inference rate (state update step size)
    alpha_trans     : transition net weight learning rate
    alpha_efe       : EFE net weight learning rate
    alpha_policy    : policy net weight learning rate
    discount        : TD discount factor γ
    obs_scale       : per-dimension scale for normalising observations
    action_candidates : discrete action values to evaluate
    N_infer         : inference-settle ticks (α=0) per online learning pass
    N_learn         : weight-update ticks (α>0) per online learning pass
    N_action        : inference ticks for EFE queries during action selection
    N_replay        : weight-update ticks per replay sample
    N_replay_query  : target query ticks during replay (< N_action for speed)
    adaptive_inference : if True, allow extra settle ticks until latent-state
                      drift falls below settle_tol. This keeps the online loop
                      closer to the fixed-point regime assumed by the PC /
                      backprop equivalence.
    settle_tol      : max absolute free-state change per tick used as the
                      convergence stopping criterion for adaptive inference.
    max_infer_ticks : cap on settle ticks for supervised PC passes.
    max_action_ticks : cap on settle ticks for live action / value queries.
    max_replay_query_ticks : cap on settle ticks for replay bootstrap queries.
    beta            : softmax inverse temperature for action selection
    buffer_size     : replay buffer capacity
    batch_size      : number of transitions sampled per replay step
    target_update_freq : EFE target network sync period (episodes)
    obs_cost_scale  : per-dimension boundary scale for log-likelihood cost.
                      If provided, uses -log p*(obs) = 0.5*sum((obs/scale)²)
                      as the pragmatic cost instead of the flat environment reward.
                      This is the active inference formulation: the agent minimises
                      expected surprise under a Gaussian goal prior centred on zero.
                      CartPole: [2.4, 3.0, 0.2095, 2.5] (termination boundaries).
    reset_efe_target_state : if True, bootstrap queries start from the PC
                      network's neutral prior instead of the previous live state.
    policy_target_clip : clip magnitude for EFE-derived policy logits.
    policy_train_every_episodes : how many complete episodes to accumulate before
                      updating the amortised policy on trajectory states.
    policy_batch_size : maximum number of trajectory states used in each policy
                      refresh, to approximate a single batched update.
    seed            : RNG seed for weight initialisation
    """

    def __init__(
        self,
        dim_obs: int,
        dim_s: int,
        dim_a: int,
        hidden_trans: int = 64,
        hidden_efe: int = 64,
        hidden_policy: int = 64,
        gamma_pc: float = 0.1,
        alpha_trans: float = 0.01,
        alpha_efe: float = 0.005,
        alpha_policy: Optional[float] = None,
        discount: float = 0.99,
        obs_scale: Optional[np.ndarray] = None,
        action_candidates: Optional[list] = None,
        N_infer: int = 20,
        N_learn: int = 10,
        N_action: int = 50,
        N_replay: int = 5,
        N_replay_query: int = 20,
        adaptive_inference: bool = True,
        settle_tol: float = 0.001,
        max_infer_ticks: Optional[int] = None,
        max_action_ticks: Optional[int] = None,
        max_replay_query_ticks: Optional[int] = None,
        beta: float = 1.0,
        buffer_size: int = 10000,
        batch_size: int = 8,
        target_update_freq: int = 50,
        obs_cost_scale: Optional[np.ndarray] = None,
        terminal_cost: Optional[float] = None,
        efe_hidden_state_clip: float = 10.0,
        efe_output_state_clip: float = 2.0,
        efe_error_clip: float = 1.0,
        efe_target_scale: Optional[float] = None,
        efe_learn_gamma: float = 0.0,
        reset_efe_target_state: bool = True,
        policy_target_clip: float = 5.0,
        policy_warmup_updates: int = 20,
        policy_fallback_on_unsettled: bool = True,
        policy_query_trust_tol: Optional[float] = None,
        policy_train_every_episodes: int = 5,
        policy_batch_size: int = 64,
        seed: int = 0,
    ):
        self.dim_obs = dim_obs
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.gamma_pc = gamma_pc
        self.alpha_trans = alpha_trans
        self.alpha_efe = alpha_efe
        self.alpha_policy = alpha_efe if alpha_policy is None else alpha_policy
        self.discount = discount
        self.N_infer = N_infer
        self.N_learn = N_learn
        self.N_action = N_action
        self.N_replay = N_replay
        self.N_replay_query = N_replay_query
        self.adaptive_inference = adaptive_inference
        self.settle_tol = settle_tol
        self.max_infer_ticks = max(N_infer, 100) if max_infer_ticks is None else max_infer_ticks
        self.max_action_ticks = max(N_action, 200) if max_action_ticks is None else max_action_ticks
        if max_replay_query_ticks is None:
            self.max_replay_query_ticks = max(N_replay_query, 100)
        else:
            self.max_replay_query_ticks = max_replay_query_ticks
        self.beta = beta
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.action_candidates = action_candidates if action_candidates is not None else [-1.0, 1.0]
        self.num_actions = len(self.action_candidates)

        self.obs_scale = (np.ones(dim_s, dtype=np.float64)
                          if obs_scale is None else np.asarray(obs_scale, dtype=np.float64))
        self.obs_cost_scale = (None if obs_cost_scale is None
                               else np.asarray(obs_cost_scale, dtype=np.float64))
        if terminal_cost is None and self.obs_cost_scale is not None:
            terminal_cost = 1.0 / (1.0 - discount)
        self.terminal_cost = terminal_cost
        self.efe_target_scale = (self.terminal_cost
                                 if efe_target_scale is None and self.terminal_cost is not None
                                 else (1.0 if efe_target_scale is None else efe_target_scale))
        self.efe_hidden_state_clip = efe_hidden_state_clip
        self.efe_output_state_clip = efe_output_state_clip
        self.efe_error_clip = efe_error_clip
        self.efe_learn_gamma = efe_learn_gamma
        self.reset_efe_target_state = reset_efe_target_state
        self.policy_target_clip = policy_target_clip
        self.policy_warmup_updates = policy_warmup_updates
        self.policy_fallback_on_unsettled = policy_fallback_on_unsettled
        self.policy_query_trust_tol = (self.settle_tol if policy_query_trust_tol is None
                                       else policy_query_trust_tol)
        self.policy_train_every_episodes = policy_train_every_episodes
        self.policy_batch_size = policy_batch_size

        # ── transition net: p(s'|s,a) ─────────────────────────────────────────
        # tanh hidden: both a=-1 and a=+1 produce non-zero activations (relu
        # zeros out a=-1, making it indistinguishable from a=+1)
        self.transition_net = PCNet3Layer(
            k_lut   = [dim_s, hidden_trans, dim_s + dim_a],
            act_lut = ['linear', 'tanh', 'linear'],
            wclip   = 20.0,
            gamma   = gamma_pc,
            alpha   = alpha_trans,
            seed    = seed,
            rtl_init    = False,
            gen_k_lut   = None,
        )

        # ── EFE net: G_ψ(s,a) scalar ─────────────────────────────────────────
        self.efe_net = PCNet3Layer(
            k_lut   = [self.num_actions, hidden_efe, dim_s],
            act_lut = ['linear', 'tanh', 'linear'],
            wclip   = 20.0,
            xclip_lut = [self.efe_output_state_clip, self.efe_hidden_state_clip, None],
            eps_clip_lut = [self.efe_error_clip, self.efe_error_clip, self.efe_error_clip],
            gamma   = gamma_pc,
            alpha   = alpha_efe,
            seed    = seed + 1,
            rtl_init    = False,
            gen_k_lut   = None,
        )

        # ── policy net: amortised action posterior π(a|s) ───────────────────
        self.policy_net = PCNet3Layer(
            k_lut   = [self.num_actions, hidden_policy, dim_s],
            act_lut = ['linear', 'tanh', 'linear'],
            wclip   = 20.0,
            xclip_lut = [policy_target_clip, 10.0, None],
            eps_clip_lut = [2.0, 2.0, 2.0],
            gamma   = gamma_pc,
            alpha   = self.alpha_policy,
            seed    = seed + 2,
            rtl_init    = False,
            gen_k_lut   = None,
        )

        # ── EFE target network (frozen copy, periodically synced) ─────────────
        self._efe_target_W0    = self.efe_net.layer0.W.copy()
        self._efe_target_bias0 = self.efe_net.layer0.bias.copy()
        self._efe_target_W1    = self.efe_net.layer1.W.copy()
        self._efe_target_bias1 = self.efe_net.layer1.bias.copy()

        # ── replay buffer ─────────────────────────────────────────────────────
        # Stores raw transitions; G_hat is RECOMPUTED at replay time from the
        # current target network so we never train against stale targets.
        self._replay_buf: collections.deque = collections.deque(maxlen=buffer_size)

        self._step_count = 0
        self._grad_steps = 0
        self._policy_updates = 0
        self._episodes_seen = 0
        self._policy_histories: list[np.ndarray] = []
        self._episode_states: list[np.ndarray] = []
        self._inference_stat_labels = (
            'efe_online_settle',
            'efe_replay_settle',
            'efe_query',
            'efe_target_query',
            'policy_query',
            'policy_settle',
        )
        self._episode_infer_stats = self._new_inference_stats()
        self._lifetime_infer_stats = self._new_inference_stats()
        self._last_policy_query_info = {
            'ticks_run': 0,
            'requested_ticks': 0,
            'final_delta': 0.0,
            'hit_max': False,
            'settled': True,
        }
        self._policy_action_fallbacks = 0
        self._policy_bootstrap_fallbacks = 0
        self._episode_policy_action_fallbacks = 0
        self._episode_policy_bootstrap_fallbacks = 0

        self.s_prev      = np.zeros(dim_s, dtype=np.float64)
        self.a_prev      = np.zeros(dim_a, dtype=np.float64)
        self.G_prev      = 0.0
        self._first_step = True

    # ── public API ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call at the start of each episode; preserves learned weights."""
        self.transition_net.reset_state()
        self.efe_net.reset_state()
        self.policy_net.reset_state()
        self.s_prev      = np.zeros(self.dim_s, dtype=np.float64)
        self.a_prev      = np.zeros(self.dim_a, dtype=np.float64)
        self.G_prev      = 0.0
        self._first_step = True
        self._episode_states = []
        self._episode_infer_stats = self._new_inference_stats()
        self._episode_policy_action_fallbacks = 0
        self._episode_policy_bootstrap_fallbacks = 0

    def step(self, obs: np.ndarray, reward: float = 0.0, done: bool = False) -> np.ndarray:
        """
        Process one environment step.

        obs    : current observation, shape (dim_obs,)
        reward : scalar reward from the previous action (0 on first call)
        done   : True if this is the terminal observation
        Returns selected action, shape (dim_a,)
        """
        raw_obs = np.asarray(obs[:self.dim_s], dtype=np.float64)
        s_t = raw_obs / self.obs_scale

        if not self._first_step:
            self._learn_transition(s_t)
            eps_trans = float(np.mean(np.abs(self.transition_net.layer0.eps)))

            if done:
                G_bootstrap = 0.0
                # Terminal penalty: mark failure as worst-case outcome.
                # Without this, terminal states (G_bootstrap=0) appear PREFERRED
                # because non-terminal states accumulate future costs > 0.
                # Set terminal cost ≈ max plausible return-to-go.
                if self.obs_cost_scale is not None:
                    cost_t = self.terminal_cost
                else:
                    cost_t = -float(reward)  # reward=0 at terminal → cost=0
            else:
                target_G = self._query_efe_target_vector(s_t)
                G_bootstrap = float(np.dot(self._policy_probs_for_bootstrap(s_t, target_G), target_G))
                if self.obs_cost_scale is not None:
                    cost_t = 0.5 * float(np.sum((raw_obs / self.obs_cost_scale) ** 2))
                else:
                    cost_t = -float(reward)
            self.G_prev = G_bootstrap

            G_hat = cost_t + eps_trans + self.discount * G_bootstrap

            # Online update for current transition
            self._learn_efe_single(self.s_prev, self._action_to_index(self.a_prev), G_hat)
            self._grad_steps += 1
            if self._grad_steps % self.target_update_freq == 0:
                self._sync_target()

            # Store transition; raw_obs kept so cost can be recomputed at replay time
            self._replay_buf.append(
                (self.s_prev.copy(), self.a_prev.copy(), self._action_to_index(self.a_prev),
                 float(reward), s_t.copy(), raw_obs.copy(), bool(done), float(eps_trans))
            )

        if done:
            self._finish_episode()
            self._first_step = True
            return np.zeros(self.dim_a, dtype=np.float64)

        G_vals = self._query_efe_values(s_t)
        a = self._select_action(s_t, G_vals)
        self._episode_states.append(s_t.copy())

        self.s_prev      = s_t.copy()
        self.a_prev      = a.copy()
        self._first_step = False
        self._step_count += 1
        return a

    # ── learning ───────────────────────────────────────────────────────────────

    def _new_inference_stats(self) -> dict[str, dict[str, float]]:
        return {
            label: {
                'calls': 0.0,
                'ticks': 0.0,
                'extra_ticks': 0.0,
                'final_delta': 0.0,
                'max_hits': 0.0,
            }
            for label in self._inference_stat_labels
        }

    def _record_inference_stat(self, label: Optional[str], ticks_run: int, n_ticks: int,
                               final_delta: float, hit_max: bool) -> None:
        if label is None:
            return
        for stats in (self._episode_infer_stats, self._lifetime_infer_stats):
            slot = stats[label]
            slot['calls'] += 1.0
            slot['ticks'] += float(ticks_run)
            slot['extra_ticks'] += float(max(0, ticks_run - n_ticks))
            slot['final_delta'] += float(final_delta)
            slot['max_hits'] += float(hit_max)

    def _run_inference_ticks(self, net: PCNet3Layer, x_top: np.ndarray,
                             y_bottom: Optional[np.ndarray], clamp_bottom: bool,
                             n_ticks: int, max_ticks: Optional[int] = None,
                             adaptive: Optional[bool] = None,
                             stat_label: Optional[str] = None) -> dict[str, float | bool]:
        """
        Run inference ticks, optionally extending beyond n_ticks until the free
        latent states stop moving.

        We use latent-state drift rather than raw eps as the stop condition
        because fixed points in free layers satisfy back_eff ≈ eps, not
        necessarily eps ≈ 0.
        """
        adaptive = self.adaptive_inference if adaptive is None else adaptive
        total_ticks = n_ticks if not adaptive else max(n_ticks, max_ticks or n_ticks)
        ticks_run = 0
        final_delta = 0.0
        for _ in range(total_ticks):
            prev_hidden = net.layer1.x_state.copy()
            prev_bottom = None if clamp_bottom else net.layer0.x_state.copy()
            net.tick(x_top, y_bottom, clamp_top=True, clamp_bottom=clamp_bottom)
            ticks_run += 1

            max_delta = float(np.max(np.abs(net.layer1.x_state - prev_hidden)))
            if prev_bottom is not None:
                max_delta = max(max_delta, float(np.max(np.abs(net.layer0.x_state - prev_bottom))))
            final_delta = max_delta

            if not adaptive or ticks_run < n_ticks:
                continue
            if max_delta <= self.settle_tol:
                break
        hit_max = bool(adaptive and ticks_run >= total_ticks and final_delta > self.settle_tol)
        self._record_inference_stat(stat_label, ticks_run, n_ticks, final_delta, hit_max)
        return {
            'ticks_run': ticks_run,
            'requested_ticks': n_ticks,
            'max_ticks': total_ticks,
            'final_delta': final_delta,
            'hit_max': hit_max,
            'settled': (final_delta <= self.settle_tol) if adaptive else True,
        }

    def _learn_transition(self, s_t: np.ndarray) -> None:
        sa = np.concatenate([self.s_prev, self.a_prev])
        self.transition_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        for _ in range(self.N_infer):
            self.transition_net.tick(sa, s_t, clamp_top=True, clamp_bottom=True)
        self.transition_net.set_rates(alpha=self.alpha_trans, gamma=self.gamma_pc)
        for _ in range(self.N_learn):
            self.transition_net.tick(sa, s_t, clamp_top=True, clamp_bottom=True)

    def _learn_efe_single(self, s: np.ndarray, action_idx: int, G_hat: float) -> None:
        """One online EFE update: settle then update the chosen action head."""
        G_arr = self._query_efe_vector(s, reset_state=True)
        G_arr[action_idx] = self._normalize_efe_value(G_hat)
        self.efe_net.reset_state()
        self.efe_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_inference_ticks(
            self.efe_net, s, G_arr, clamp_bottom=True,
            n_ticks=self.N_infer, max_ticks=self.max_infer_ticks,
            stat_label='efe_online_settle',
        )
        # Keep the inferred hidden state fixed during EFE weight updates.
        # This preserves the settle-then-learn schedule and avoids target-driven
        # state drift when large terminal costs enter the replay stream.
        self.efe_net.set_rates(alpha=self.alpha_efe, gamma=self.efe_learn_gamma)
        for _ in range(self.N_learn):
            self.efe_net.tick(s, G_arr, clamp_top=True, clamp_bottom=True)

    def _replay_efe(self) -> None:
        """
        Sample a batch from replay buffer and update EFE weights.

        G_hat is RECOMPUTED from the current target network (N_replay_query ticks)
        rather than using the stored value — prevents training against stale targets.
        """
        n = len(self._replay_buf)
        if n < max(4, self.batch_size):
            return
        idxs = np.random.choice(n, self.batch_size, replace=False)
        for i in idxs:
            s, a, action_idx, reward, s_next, raw_next, is_done, eps_trans = self._replay_buf[i]
            if is_done:
                G_boot = 0.0
                cost = (self.terminal_cost if self.obs_cost_scale is not None
                        else -reward)
            else:
                target_G = self._query_efe_target_vector(s_next, n_ticks=self.N_replay_query)
                G_boot = float(np.dot(self._policy_probs_for_bootstrap(s_next, target_G), target_G))
                cost = (0.5 * float(np.sum((raw_next / self.obs_cost_scale) ** 2))
                        if self.obs_cost_scale is not None else -reward)
            G_hat = cost + eps_trans + self.discount * G_boot
            G_arr = self._query_efe_vector(s, reset_state=True)
            G_arr[action_idx] = self._normalize_efe_value(G_hat)
            # Settle phase: bring hidden layer to correct state for this (sa, G_hat)
            # pair before weight updates.  Without this, Hebbian updates use the
            # hidden state left over from the previous sample → corrupted gradients.
            self.efe_net.reset_state()
            self.efe_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
            self._run_inference_ticks(
                self.efe_net, s, G_arr, clamp_bottom=True,
                n_ticks=self.N_infer, max_ticks=self.max_infer_ticks,
                stat_label='efe_replay_settle',
            )
            self.efe_net.set_rates(alpha=self.alpha_efe, gamma=self.efe_learn_gamma)
            for _ in range(self.N_replay):
                self.efe_net.tick(s, G_arr, clamp_top=True, clamp_bottom=True)

    # ── action selection ───────────────────────────────────────────────────────

    def _select_action(self, s_t: np.ndarray, G_vals: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample an action from the amortised policy network once it has seen
        enough EFE supervision; otherwise fall back to the direct EFE posterior.
        """
        use_policy = self._policy_updates >= self.policy_warmup_updates
        if self._policy_updates >= self.policy_warmup_updates:
            logits = self._query_policy_logits(s_t)
            if self.policy_fallback_on_unsettled and not self._policy_query_is_trusted():
                use_policy = False
        else:
            use_policy = False
        if not use_policy:
            if G_vals is None:
                G_vals = self._query_efe_values(s_t)
            logits = self._policy_logits_from_efe(G_vals)
            if self._policy_updates >= self.policy_warmup_updates:
                self._policy_action_fallbacks += 1
                self._episode_policy_action_fallbacks += 1
        return np.full(self.dim_a, self.action_candidates[self._sample_index_from_logits(logits)],
                       dtype=np.float64)

    def _finish_episode(self) -> None:
        """Run end-of-episode updates following the Millidge-style schedule."""
        self._replay_efe()
        if self._episode_states:
            self._policy_histories.append(np.asarray(self._episode_states, dtype=np.float64))
        self._episode_states = []
        self._episodes_seen += 1
        if self._episodes_seen % self.policy_train_every_episodes == 0 and self._policy_histories:
            self._learn_policy_from_histories()
            self._policy_histories = []
        if self._episodes_seen % self.target_update_freq == 0:
            self._sync_target()

    def _learn_policy_single(self, s_t: np.ndarray, target_logits: np.ndarray) -> None:
        """Train the PC policy net to amortise EFE-implied action preferences."""
        self.policy_net.reset_state()
        self.policy_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_inference_ticks(
            self.policy_net, s_t, target_logits, clamp_bottom=True,
            n_ticks=self.N_infer, max_ticks=self.max_infer_ticks,
            stat_label='policy_settle',
        )
        self.policy_net.set_rates(alpha=self.alpha_policy, gamma=0.0)
        for _ in range(self.N_learn):
            self.policy_net.tick(s_t, target_logits, clamp_top=True, clamp_bottom=True)
        self._policy_updates += 1

    def _learn_policy_from_histories(self) -> None:
        """Approximate Millidge's history-based policy update using PC supervision."""
        states = np.concatenate(self._policy_histories, axis=0)
        if len(states) > self.policy_batch_size:
            idxs = np.random.choice(len(states), self.policy_batch_size, replace=False)
            states = states[idxs]
        for s_t in states:
            G_vals = self._query_efe_values(s_t)
            self._learn_policy_single(s_t, self._policy_logits_from_efe(G_vals))

    def _query_policy_logits(self, s_t: np.ndarray) -> np.ndarray:
        """Query the policy network without contaminating its latent state."""
        return self._query_policy_logits_with_ticks(s_t, n_ticks=self.N_action)

    def _query_policy_logits_with_ticks(self, s_t: np.ndarray, n_ticks: int,
                                        adaptive: Optional[bool] = None) -> np.ndarray:
        """Query the policy network with an explicit inference budget."""
        adaptive = self.adaptive_inference if adaptive is None else adaptive
        l0_x = self.policy_net.layer0.x_state.copy()
        l1_x = self.policy_net.layer1.x_state.copy()
        l2_x = self.policy_net.layer2.x_state.copy()
        self.policy_net.reset_state()
        self.policy_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        query_info = self._run_inference_ticks(
            self.policy_net, s_t, y_bottom=None, clamp_bottom=False,
            n_ticks=n_ticks, max_ticks=self.max_action_ticks, adaptive=adaptive,
            stat_label='policy_query' if adaptive else None,
        )
        self._last_policy_query_info = dict(query_info)
        logits = self.policy_net.x0
        self.policy_net.layer0.x_state[:] = l0_x
        self.policy_net.layer1.x_state[:] = l1_x
        self.policy_net.layer2.x_state[:] = l2_x
        return logits

    def _query_efe_values(self, s_t: np.ndarray) -> np.ndarray:
        """Evaluate live vector EFE values and return them in raw cost units."""
        return self._denormalize_efe_value(self._query_efe_vector(s_t, reset_state=False))

    def _query_efe_vector(self, s_t: np.ndarray, reset_state: bool = False,
                          n_ticks: Optional[int] = None,
                          adaptive: Optional[bool] = None) -> np.ndarray:
        """Query the live EFE vector in internal normalised units."""
        n_ticks = self.N_action if n_ticks is None else n_ticks
        adaptive = self.adaptive_inference if adaptive is None else adaptive
        l0_x = self.efe_net.layer0.x_state.copy()
        l1_x = self.efe_net.layer1.x_state.copy()
        l2_x = self.efe_net.layer2.x_state.copy()
        if reset_state:
            self.efe_net.reset_state()
        self.efe_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        max_ticks = self.max_action_ticks if n_ticks >= self.N_action else self.max_replay_query_ticks
        self._run_inference_ticks(
            self.efe_net, s_t, y_bottom=None, clamp_bottom=False,
            n_ticks=n_ticks, max_ticks=max_ticks, adaptive=adaptive,
            stat_label='efe_query' if adaptive else None,
        )
        G_vec = self.efe_net.x0
        self.efe_net.layer0.x_state[:] = l0_x
        self.efe_net.layer1.x_state[:] = l1_x
        self.efe_net.layer2.x_state[:] = l2_x
        return G_vec

    # ── EFE target network ─────────────────────────────────────────────────────

    def _sync_target(self) -> None:
        self._efe_target_W0    = self.efe_net.layer0.W.copy()
        self._efe_target_bias0 = self.efe_net.layer0.bias.copy()
        self._efe_target_W1    = self.efe_net.layer1.W.copy()
        self._efe_target_bias1 = self.efe_net.layer1.bias.copy()

    def _query_efe_target_vector(self, s_t: np.ndarray,
                                 n_ticks: Optional[int] = None,
                                 adaptive: Optional[bool] = None) -> np.ndarray:
        """
        Query the frozen target EFE network.
        Saves and restores both weights and x_states to avoid contaminating the
        live network's state during bootstrap and replay queries.

        Optionally resets the latent state before settling so bootstrap values
        are read from a neutral prior rather than whichever trajectory or replay
        sample happened to be processed immediately beforehand.
        """
        n_ticks = n_ticks if n_ticks is not None else self.N_action
        adaptive = self.adaptive_inference if adaptive is None else adaptive

        l0_x = self.efe_net.layer0.x_state.copy()
        l1_x = self.efe_net.layer1.x_state.copy()
        l2_x = self.efe_net.layer2.x_state.copy()
        W0_live, b0_live = self.efe_net.layer0.W, self.efe_net.layer0.bias
        W1_live, b1_live = self.efe_net.layer1.W, self.efe_net.layer1.bias

        self.efe_net.layer0.W    = self._efe_target_W0
        self.efe_net.layer0.bias = self._efe_target_bias0
        self.efe_net.layer1.W    = self._efe_target_W1
        self.efe_net.layer1.bias = self._efe_target_bias1
        if self.reset_efe_target_state:
            self.efe_net.reset_state()

        self.efe_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        max_ticks = self.max_action_ticks if n_ticks >= self.N_action else self.max_replay_query_ticks
        self._run_inference_ticks(
            self.efe_net, s_t, y_bottom=None, clamp_bottom=False,
            n_ticks=n_ticks, max_ticks=max_ticks, adaptive=adaptive,
            stat_label='efe_target_query' if adaptive else None,
        )
        G = self._denormalize_efe_value(self.efe_net.x0)

        self.efe_net.layer0.W    = W0_live
        self.efe_net.layer0.bias = b0_live
        self.efe_net.layer1.W    = W1_live
        self.efe_net.layer1.bias = b1_live
        self.efe_net.layer0.x_state[:] = l0_x
        self.efe_net.layer1.x_state[:] = l1_x
        self.efe_net.layer2.x_state[:] = l2_x

        return G

    def _query_efe_target(self, s_t: np.ndarray, a: np.ndarray,
                           n_ticks: Optional[int] = None) -> float:
        """Compatibility wrapper: return the chosen action's target EFE."""
        return float(self._query_efe_target_vector(s_t, n_ticks=n_ticks)[self._action_to_index(a)])

    def _query_efe(self, s_t: np.ndarray, a: np.ndarray) -> float:
        """Compatibility wrapper: return the chosen action's live EFE."""
        return float(self._query_efe_values(s_t)[self._action_to_index(a)])

    def _normalize_efe_value(self, G: float) -> float:
        return np.asarray(G, dtype=np.float64) / self.efe_target_scale

    def _denormalize_efe_value(self, g_norm: float) -> float:
        return np.asarray(g_norm, dtype=np.float64) * self.efe_target_scale

    def _policy_logits_from_efe(self, G_vals: np.ndarray) -> np.ndarray:
        """
        Convert EFE costs into centred action logits.
        Softmax is invariant to additive constants, so removing the mean strips
        the large common offset that otherwise dominates policy learning.
        """
        logits = -(np.asarray(G_vals, dtype=np.float64) - np.mean(G_vals))
        return np.clip(logits, -self.policy_target_clip, self.policy_target_clip)

    def _sample_index_from_logits(self, logits: np.ndarray) -> int:
        scaled = self.beta * np.asarray(logits, dtype=np.float64)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(np.random.choice(len(self.action_candidates), p=probs))

    def _policy_probs_for_bootstrap(self, s_t: np.ndarray, G_vals: np.ndarray) -> np.ndarray:
        """
        Use the amortised policy once it has been trained; otherwise bootstrap
        with the direct EFE posterior implied by the current value landscape.
        """
        use_policy = self._policy_updates >= self.policy_warmup_updates
        if self._policy_updates >= self.policy_warmup_updates:
            logits = self._query_policy_logits(s_t)
            if self.policy_fallback_on_unsettled and not self._policy_query_is_trusted():
                use_policy = False
        else:
            use_policy = False
        if not use_policy:
            logits = self._policy_logits_from_efe(G_vals)
            if self._policy_updates >= self.policy_warmup_updates:
                self._policy_bootstrap_fallbacks += 1
                self._episode_policy_bootstrap_fallbacks += 1
        scaled = self.beta * np.asarray(logits, dtype=np.float64)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return probs

    def _policy_query_is_trusted(self) -> bool:
        info = self._last_policy_query_info
        return (not bool(info['hit_max'])
                and float(info['final_delta']) <= self.policy_query_trust_tol)

    def _action_to_index(self, a: np.ndarray) -> int:
        """Map the continuous action representation back onto the discrete set."""
        a_scalar = float(np.asarray(a, dtype=np.float64).reshape(-1)[0])
        diffs = [abs(a_scalar - float(candidate)) for candidate in self.action_candidates]
        return int(np.argmin(diffs))

    # ── public diagnostics ─────────────────────────────────────────────────────

    def efe_values_for_state(self, obs: np.ndarray) -> np.ndarray:
        """Return raw EFE costs for each discrete action at the given state."""
        s_t = np.asarray(obs[:self.dim_s], dtype=np.float64) / self.obs_scale
        return self._query_efe_values(s_t)

    def policy_logits_for_state(self, obs: np.ndarray) -> np.ndarray:
        """Return the policy network's logits for each discrete action."""
        s_t = np.asarray(obs[:self.dim_s], dtype=np.float64) / self.obs_scale
        return self._query_policy_logits(s_t)

    def policy_probs_for_state(self, obs: np.ndarray) -> np.ndarray:
        """Return the amortised policy probabilities for each discrete action."""
        logits = self.policy_logits_for_state(obs)
        scaled = self.beta * np.asarray(logits, dtype=np.float64)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return probs

    def bootstrap_policy_probs_for_state(self, obs: np.ndarray) -> np.ndarray:
        """
        Return the policy probabilities used in the EFE bootstrap target.
        Before the policy warmup finishes, this is the direct EFE posterior.
        """
        s_t = np.asarray(obs[:self.dim_s], dtype=np.float64) / self.obs_scale
        G_vals = self._query_efe_target_vector(s_t)
        return self._policy_probs_for_bootstrap(s_t, G_vals)

    def convergence_profile_for_state(self, obs: np.ndarray,
                                      ticks: list[int] | tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200)
                                      ) -> dict:
        """
        Probe how far the agent is from fixed-point inference as the tick budget
        increases. Large movement past the default inference budget means the
        outer loop is operating outside the regime assumed by the PC/backprop
        equivalence.
        """
        s_t = np.asarray(obs[:self.dim_s], dtype=np.float64) / self.obs_scale
        tick_list = sorted(set(int(t) for t in ticks))
        efe_values = []
        policy_logits = []
        for n_ticks in tick_list:
            efe_values.append(self._denormalize_efe_value(
                self._query_efe_vector(s_t, reset_state=True, n_ticks=n_ticks, adaptive=False)
            ))
            policy_logits.append(self._query_policy_logits_with_ticks(s_t, n_ticks=n_ticks, adaptive=False))
        return {
            'ticks': np.asarray(tick_list, dtype=np.int64),
            'efe_values': np.asarray(efe_values, dtype=np.float64),
            'policy_logits': np.asarray(policy_logits, dtype=np.float64),
        }

    # ── diagnostics ────────────────────────────────────────────────────────────

    @property
    def transition_prediction_error(self) -> float:
        return float(np.mean(np.abs(self.transition_net.layer0.eps)))

    @property
    def G_estimate(self) -> float:
        return self.G_prev

    def _summarize_inference_stats(self, stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for label, slot in stats.items():
            calls = float(slot['calls'])
            if calls <= 0.0:
                summary[label] = {
                    'calls': 0.0,
                    'avg_ticks': 0.0,
                    'avg_extra_ticks': 0.0,
                    'avg_final_delta': 0.0,
                    'max_hit_rate': 0.0,
                }
            else:
                summary[label] = {
                    'calls': calls,
                    'avg_ticks': float(slot['ticks'] / calls),
                    'avg_extra_ticks': float(slot['extra_ticks'] / calls),
                    'avg_final_delta': float(slot['final_delta'] / calls),
                    'max_hit_rate': float(slot['max_hits'] / calls),
                }
        return summary

    @property
    def inference_diagnostics(self) -> dict[str, dict[str, float]]:
        return self._summarize_inference_stats(self._episode_infer_stats)

    @property
    def lifetime_inference_diagnostics(self) -> dict[str, dict[str, float]]:
        return self._summarize_inference_stats(self._lifetime_infer_stats)

    @property
    def policy_update_count(self) -> int:
        return self._policy_updates

    @property
    def last_policy_query_info(self) -> dict[str, float | bool]:
        return dict(self._last_policy_query_info)

    @property
    def policy_action_fallback_count(self) -> int:
        return self._policy_action_fallbacks

    @property
    def policy_bootstrap_fallback_count(self) -> int:
        return self._policy_bootstrap_fallbacks

    @property
    def episode_policy_action_fallback_count(self) -> int:
        return self._episode_policy_action_fallbacks

    @property
    def episode_policy_bootstrap_fallback_count(self) -> int:
        return self._episode_policy_bootstrap_fallbacks

    @property
    def episodes_seen(self) -> int:
        return self._episodes_seen
