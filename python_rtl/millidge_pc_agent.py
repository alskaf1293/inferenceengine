"""
millidge_pc_agent.py — Literal Millidge-style CartPole baseline with PC nets.

This branch tries to follow DeepActiveInference/active_inference.jl as closely
as possible while replacing the backprop MLPs with predictive-coding networks.

Key choices copied from the Julia baseline:
  - raw CartPole reward as the value target
  - action selection from a separate policy network
  - replay batch once per episode
  - policy/history refresh every 5 episodes
  - frozen target value network synced every 50 episodes

PC-specific concessions:
  - finite settle / learn ticks instead of Adam updates
  - adaptive settling to move closer to the fixed-point regime
  - bounded targets for numerical stability
"""
from __future__ import annotations

import collections
from typing import Optional

import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from pc_network import PCNet3Layer


class MillidgePCAgent:
    """Millidge baseline with value/policy MLPs replaced by PC networks."""

    def __init__(
        self,
        dim_s: int = 4,
        num_actions: int = 2,
        hidden: int = 100,
        discount: float = 0.99,
        gamma_pc: float = 0.1,
        alpha_value: float = 0.001,
        alpha_policy: float = 0.001,
        obs_scale: Optional[np.ndarray] = None,
        N_infer: int = 50,
        N_learn: int = 10,
        N_action: int = 100,
        N_replay_query: int = 100,
        adaptive_inference: bool = True,
        settle_tol: float = 0.001,
        max_infer_ticks: int = 200,
        max_action_ticks: int = 300,
        max_replay_query_ticks: int = 200,
        buffer_size: int = 100000,
        batch_size: int = 200,
        policy_train_every_episodes: int = 5,
        target_update_episodes: int = 50,
        replay_updates_per_episode: int = 1,
        history_passes: int = 1,
        beta: float = 1.0,
        value_target_scale: float = 100.0,
        value_target_clip: float = 5.0,
        policy_target_clip: float = 5.0,
        policy_smoothing: float = 0.02,
        policy_target_mode: str = 'greedy',
        seed: int = 0,
    ):
        self.dim_s = dim_s
        self.num_actions = num_actions
        self.discount = discount
        self.gamma_pc = gamma_pc
        self.alpha_value = alpha_value
        self.alpha_policy = alpha_policy
        self.obs_scale = (np.ones(dim_s, dtype=np.float64)
                          if obs_scale is None else np.asarray(obs_scale, dtype=np.float64))
        self.N_infer = N_infer
        self.N_learn = N_learn
        self.N_action = N_action
        self.N_replay_query = N_replay_query
        self.adaptive_inference = adaptive_inference
        self.settle_tol = settle_tol
        self.max_infer_ticks = max_infer_ticks
        self.max_action_ticks = max_action_ticks
        self.max_replay_query_ticks = max_replay_query_ticks
        self.batch_size = batch_size
        self.policy_train_every_episodes = policy_train_every_episodes
        self.target_update_episodes = target_update_episodes
        self.replay_updates_per_episode = replay_updates_per_episode
        self.history_passes = history_passes
        self.beta = beta
        self.value_target_scale = value_target_scale
        self.value_target_clip = value_target_clip
        self.policy_target_clip = policy_target_clip
        self.policy_smoothing = policy_smoothing
        self.policy_target_mode = policy_target_mode

        self.value_net = PCNet3Layer(
            k_lut=[num_actions, hidden, dim_s],
            act_lut=['linear', 'relu', 'linear'],
            wclip=20.0,
            xclip_lut=[value_target_clip, 10.0, None],
            eps_clip_lut=[1.0, 1.0, 1.0],
            gamma=gamma_pc,
            alpha=alpha_value,
            seed=seed,
            rtl_init=False,
            gen_k_lut=None,
        )
        self.policy_net = PCNet3Layer(
            k_lut=[num_actions, hidden, dim_s],
            act_lut=['linear', 'relu', 'linear'],
            wclip=20.0,
            xclip_lut=[policy_target_clip, 10.0, None],
            eps_clip_lut=[1.0, 1.0, 1.0],
            gamma=gamma_pc,
            alpha=alpha_policy,
            seed=seed + 1,
            rtl_init=False,
            gen_k_lut=None,
        )

        self._target_W0 = self.value_net.layer0.W.copy()
        self._target_b0 = self.value_net.layer0.bias.copy()
        self._target_W1 = self.value_net.layer1.W.copy()
        self._target_b1 = self.value_net.layer1.bias.copy()

        self._memory: collections.deque = collections.deque(maxlen=buffer_size)
        self._histories: list[np.ndarray] = []
        self._episode_states: list[np.ndarray] = []
        self._episodes_seen = 0
        self._last_value_loss = 0.0

        self._prev_state: Optional[np.ndarray] = None
        self._prev_action_idx: Optional[int] = None

    def reset(self) -> None:
        self.value_net.reset_state()
        self.policy_net.reset_state()
        self._episode_states = []
        self._prev_state = None
        self._prev_action_idx = None

    def step(self, obs: np.ndarray, reward: float = 0.0, done: bool = False) -> int:
        state = np.asarray(obs[:self.dim_s], dtype=np.float64) / self.obs_scale

        if self._prev_state is not None and self._prev_action_idx is not None:
            self._memory.append((
                self._prev_state.copy(),
                int(self._prev_action_idx),
                float(reward),
                state.copy(),
                bool(done),
            ))

        if done:
            self._finish_episode()
            self._prev_state = None
            self._prev_action_idx = None
            return 0

        self._episode_states.append(state.copy())
        logits = self._query_policy_logits(state)
        action_idx = self._sample_index_from_logits(logits)
        self._prev_state = state.copy()
        self._prev_action_idx = action_idx
        return action_idx

    def _run_inference_ticks(self, net: PCNet3Layer, x_top: np.ndarray,
                             y_bottom: Optional[np.ndarray], clamp_bottom: bool,
                             n_ticks: int, max_ticks: int) -> None:
        total_ticks = n_ticks if not self.adaptive_inference else max(n_ticks, max_ticks)
        for ticks_run in range(total_ticks):
            prev_hidden = net.layer1.x_state.copy()
            prev_bottom = None if clamp_bottom else net.layer0.x_state.copy()
            net.tick(x_top, y_bottom, clamp_top=True, clamp_bottom=clamp_bottom)
            if not self.adaptive_inference or ticks_run + 1 < n_ticks:
                continue
            max_delta = float(np.max(np.abs(net.layer1.x_state - prev_hidden)))
            if prev_bottom is not None:
                max_delta = max(max_delta, float(np.max(np.abs(net.layer0.x_state - prev_bottom))))
            if max_delta <= self.settle_tol:
                break

    def _finish_episode(self) -> None:
        for _ in range(self.replay_updates_per_episode):
            self._replay_value_once()
        if self._episode_states:
            self._histories.append(np.asarray(self._episode_states, dtype=np.float64))
        self._episode_states = []
        self._episodes_seen += 1
        if self._episodes_seen % self.policy_train_every_episodes == 0 and self._histories:
            for _ in range(self.history_passes):
                self._history_policy_refresh()
            self._histories = []
        if self._episodes_seen % self.target_update_episodes == 0:
            self._sync_target()

    def _replay_value_once(self) -> None:
        if not self._memory:
            return
        batch_size = min(self.batch_size, len(self._memory))
        idxs = np.random.choice(len(self._memory), batch_size, replace=False)
        losses = []
        for idx in idxs:
            state, action_idx, reward, next_state, done = self._memory[idx]
            target = float(reward)
            if not done:
                next_policy = self._policy_probs(next_state)
                next_value = self._query_target_value(next_state, n_ticks=self.N_replay_query)
                target += self.discount * float(np.dot(next_policy, next_value))
            current = self._query_value_raw(state, reset_state=True)
            current[action_idx] = target
            losses.append(abs(target - self._query_value_raw(state, reset_state=True)[action_idx]))
            self._learn_value_vector(state, current)
        self._last_value_loss = float(np.mean(losses)) if losses else 0.0

    def _history_policy_refresh(self) -> None:
        states = np.concatenate(self._histories, axis=0)
        for state in states:
            values = self._query_value_raw(state)
            self._learn_policy_logits(state, self._policy_target_from_value(values))
            policy_probs = self._policy_probs(state)
            value_target = self._value_logits_from_probs(policy_probs)
            self._learn_value_vector(state, value_target, target_is_logits=True)

    def _learn_value_vector(self, state: np.ndarray, target: np.ndarray,
                            target_is_logits: bool = False) -> None:
        if target_is_logits:
            target_vec = np.asarray(target, dtype=np.float64)
        else:
            target_vec = self._normalize_value(np.asarray(target, dtype=np.float64))
        self.value_net.reset_state()
        self.value_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_inference_ticks(
            self.value_net, state, target_vec, clamp_bottom=True,
            n_ticks=self.N_infer, max_ticks=self.max_infer_ticks,
        )
        self.value_net.set_rates(alpha=self.alpha_value, gamma=0.0)
        for _ in range(self.N_learn):
            self.value_net.tick(state, target_vec, clamp_top=True, clamp_bottom=True)

    def _learn_policy_logits(self, state: np.ndarray, target_logits: np.ndarray) -> None:
        self.policy_net.reset_state()
        self.policy_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_inference_ticks(
            self.policy_net, state, target_logits, clamp_bottom=True,
            n_ticks=self.N_infer, max_ticks=self.max_infer_ticks,
        )
        self.policy_net.set_rates(alpha=self.alpha_policy, gamma=0.0)
        for _ in range(self.N_learn):
            self.policy_net.tick(state, target_logits, clamp_top=True, clamp_bottom=True)

    def _query_value_raw(self, state: np.ndarray, reset_state: bool = False,
                         n_ticks: Optional[int] = None) -> np.ndarray:
        n_ticks = self.N_action if n_ticks is None else n_ticks
        l0_x = self.value_net.layer0.x_state.copy()
        l1_x = self.value_net.layer1.x_state.copy()
        l2_x = self.value_net.layer2.x_state.copy()
        if reset_state:
            self.value_net.reset_state()
        self.value_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_inference_ticks(
            self.value_net, state, y_bottom=None, clamp_bottom=False,
            n_ticks=n_ticks,
            max_ticks=self.max_action_ticks if n_ticks >= self.N_action else self.max_replay_query_ticks,
        )
        out = self._denormalize_value(self.value_net.x0)
        self.value_net.layer0.x_state[:] = l0_x
        self.value_net.layer1.x_state[:] = l1_x
        self.value_net.layer2.x_state[:] = l2_x
        return out

    def _query_target_value(self, state: np.ndarray, n_ticks: Optional[int] = None) -> np.ndarray:
        n_ticks = self.N_action if n_ticks is None else n_ticks
        l0_x = self.value_net.layer0.x_state.copy()
        l1_x = self.value_net.layer1.x_state.copy()
        l2_x = self.value_net.layer2.x_state.copy()
        W0_live, b0_live = self.value_net.layer0.W, self.value_net.layer0.bias
        W1_live, b1_live = self.value_net.layer1.W, self.value_net.layer1.bias
        self.value_net.layer0.W = self._target_W0
        self.value_net.layer0.bias = self._target_b0
        self.value_net.layer1.W = self._target_W1
        self.value_net.layer1.bias = self._target_b1
        self.value_net.reset_state()
        self.value_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_inference_ticks(
            self.value_net, state, y_bottom=None, clamp_bottom=False,
            n_ticks=n_ticks,
            max_ticks=self.max_action_ticks if n_ticks >= self.N_action else self.max_replay_query_ticks,
        )
        out = self._denormalize_value(self.value_net.x0)
        self.value_net.layer0.W = W0_live
        self.value_net.layer0.bias = b0_live
        self.value_net.layer1.W = W1_live
        self.value_net.layer1.bias = b1_live
        self.value_net.layer0.x_state[:] = l0_x
        self.value_net.layer1.x_state[:] = l1_x
        self.value_net.layer2.x_state[:] = l2_x
        return out

    def _query_policy_logits(self, state: np.ndarray, n_ticks: Optional[int] = None) -> np.ndarray:
        n_ticks = self.N_action if n_ticks is None else n_ticks
        l0_x = self.policy_net.layer0.x_state.copy()
        l1_x = self.policy_net.layer1.x_state.copy()
        l2_x = self.policy_net.layer2.x_state.copy()
        self.policy_net.reset_state()
        self.policy_net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_inference_ticks(
            self.policy_net, state, y_bottom=None, clamp_bottom=False,
            n_ticks=n_ticks, max_ticks=self.max_action_ticks,
        )
        logits = self.policy_net.x0
        self.policy_net.layer0.x_state[:] = l0_x
        self.policy_net.layer1.x_state[:] = l1_x
        self.policy_net.layer2.x_state[:] = l2_x
        return logits

    def _policy_probs(self, state: np.ndarray) -> np.ndarray:
        logits = self._query_policy_logits(state)
        return self._softmax(logits)

    def _sync_target(self) -> None:
        self._target_W0 = self.value_net.layer0.W.copy()
        self._target_b0 = self.value_net.layer0.bias.copy()
        self._target_W1 = self.value_net.layer1.W.copy()
        self._target_b1 = self.value_net.layer1.bias.copy()

    def _normalize_value(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, dtype=np.float64) / self.value_target_scale

    def _denormalize_value(self, normed: np.ndarray) -> np.ndarray:
        return np.asarray(normed, dtype=np.float64) * self.value_target_scale

    def _value_logits_from_probs(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        logits = np.log(np.clip(probs, 1e-6, 1.0))
        logits -= np.mean(logits)
        return np.clip(logits, -self.value_target_clip, self.value_target_clip)

    def _policy_target_from_value(self, values: np.ndarray) -> np.ndarray:
        if self.policy_target_mode == 'value_logits':
            centered = np.asarray(values, dtype=np.float64) - np.mean(values)
            return np.clip(centered, -self.policy_target_clip, self.policy_target_clip)
        if self.policy_target_mode != 'greedy':
            raise ValueError(f'Unsupported policy_target_mode: {self.policy_target_mode}')
        greedy = int(np.argmax(values))
        probs = np.full(self.num_actions,
                        self.policy_smoothing / max(1, self.num_actions - 1),
                        dtype=np.float64)
        probs[greedy] = 1.0 - self.policy_smoothing
        logits = np.log(np.clip(probs, 1e-6, 1.0))
        logits -= np.mean(logits)
        return np.clip(logits, -self.policy_target_clip, self.policy_target_clip)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.beta * np.asarray(logits, dtype=np.float64)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return probs

    def _sample_index_from_logits(self, logits: np.ndarray) -> int:
        probs = self._softmax(logits)
        return int(np.random.choice(self.num_actions, p=probs))

    @property
    def last_value_loss(self) -> float:
        return self._last_value_loss

    @property
    def episodes_seen(self) -> int:
        return self._episodes_seen
