"""
run_cartpole_millidge_hybrid.py — Millidge-style CartPole with swappable
backprop / predictive-coding value and policy networks.

This lets us replace the original backprop networks one by one while keeping
the outer Millidge training loop fixed.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
except ImportError:
    import gym

import sys
sys.path.insert(0, os.path.dirname(__file__))
from pc_network import PCNet3Layer


MEM_SIZE = 100000
BATCH_SIZE = 200
STATE_SIZE = 4
ACTION_SIZE = 2


@dataclass
class History:
    nS: int
    nA: int
    gamma: float
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


class BPMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_action_from_probs(probs: torch.Tensor) -> int:
    return int(torch.multinomial(probs, num_samples=1).item())


def softmax_np(logits: np.ndarray, beta: float = 1.0) -> np.ndarray:
    z = beta * np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


class BPValueModel:
    def __init__(self, hidden: int, lr: float, discount: float, seed: int, device: torch.device):
        set_seed(seed)
        self.device = device
        self.discount = discount
        self.net = BPMLP(STATE_SIZE, hidden, ACTION_SIZE).to(device)
        self.target_net = BPMLP(STATE_SIZE, hidden, ACTION_SIZE).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.last_loss = 0.0

    def predict_np(self, states: np.ndarray, target: bool = False) -> np.ndarray:
        model = self.target_net if target else self.net
        x = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            y = model(x).cpu().numpy()
        return y

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())

    def replay_update(self, memory: list[tuple[np.ndarray, int, float, np.ndarray, bool]],
                      policy_model: 'PolicyModel') -> float:
        if not memory:
            return 0.0
        batch_size = min(BATCH_SIZE, len(memory))
        minibatch = random.sample(memory, batch_size)
        states = np.stack([item[0] for item in minibatch]).astype(np.float32)
        next_states = np.stack([item[3] for item in minibatch]).astype(np.float32)
        actions = np.asarray([item[1] for item in minibatch], dtype=np.int64)
        rewards = np.asarray([item[2] for item in minibatch], dtype=np.float32)
        dones = np.asarray([item[4] for item in minibatch], dtype=np.bool_)

        x = torch.as_tensor(states, device=self.device)
        with torch.no_grad():
            next_policy = policy_model.probs_np(next_states)
            next_values = self.predict_np(next_states, target=True)
            targets = rewards + (~dones).astype(np.float32) * self.discount * np.sum(next_policy * next_values, axis=1)
            y = self.predict_np(states, target=False)
            y[np.arange(batch_size), actions] = targets
            y_t = torch.as_tensor(y, device=self.device)

        qhats = self.net(x)
        loss = F.mse_loss(qhats, y_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss


class PCValueModel:
    def __init__(
        self,
        hidden: int,
        lr: float,
        discount: float,
        seed: int,
        gamma_pc: float,
        n_infer: int,
        n_learn: int,
        n_query: int,
        adaptive_inference: bool,
        settle_tol: float,
        max_infer_ticks: int,
        max_query_ticks: int,
        value_scale: float = 100.0,
        value_clip: Optional[float] = None,
        hidden_clip: Optional[float] = None,
        eps_clip: Optional[float] = None,
        init: str = 'from_bp',
        device: Optional[torch.device] = None,
        optimizer: str = 'adam',
        trace_scale: float = 1.0,
    ):
        set_seed(seed)
        self.discount = discount
        self.lr = lr
        self.gamma_pc = gamma_pc
        self.n_infer = n_infer
        self.n_learn = n_learn
        self.n_query = n_query
        self.adaptive_inference = adaptive_inference
        self.settle_tol = settle_tol
        self.max_infer_ticks = max_infer_ticks
        self.max_query_ticks = max_query_ticks
        self.value_scale = value_scale
        self.optimizer = optimizer
        self.trace_scale = trace_scale
        self.device = device if device is not None else torch.device('cpu')
        self.net = PCNet3Layer(
            k_lut=[ACTION_SIZE, hidden, STATE_SIZE],
            # PCNet applies each layer activation to inputs from the layer above.
            # This corresponds to BP's Linear(input)->ReLU(hidden)->Linear(output).
            act_lut=['relu', 'linear', 'linear'],
            wclip=20.0,
            xclip_lut=[value_clip, hidden_clip, None],
            eps_clip_lut=[eps_clip, eps_clip, eps_clip],
            gamma=gamma_pc,
            alpha=lr,
            seed=seed,
            rtl_init=False,
            gen_k_lut=None,
        )
        self._init_weights(init, seed, hidden, device)
        self._target_W0 = self.net.layer0.W.copy()
        self._target_b0 = self.net.layer0.bias.copy()
        self._target_W1 = self.net.layer1.W.copy()
        self._target_b1 = self.net.layer1.bias.copy()
        self.last_loss = 0.0
        self.settle_calls = 0
        self.settle_ticks = 0
        self.settle_cap_hits = 0
        self.settle_final_delta = 0.0
        self._adam_t = 0
        self._adam_m = {
            'W0': np.zeros((ACTION_SIZE, hidden), dtype=np.float64),
            'b0': np.zeros(ACTION_SIZE, dtype=np.float64),
            'W1': np.zeros((hidden, STATE_SIZE), dtype=np.float64),
            'b1': np.zeros(hidden, dtype=np.float64),
        }
        self._adam_v = {k: np.zeros_like(v) for k, v in self._adam_m.items()}

    def _init_weights(self, init: str, seed: int, hidden: int, device: Optional[torch.device]) -> None:
        if init == 'pc':
            return
        if init == 'from_bp':
            bp = BPMLP(STATE_SIZE, hidden, ACTION_SIZE)
            if device is not None:
                bp = bp.to(device)
            with torch.no_grad():
                self.net.layer1.W = bp.fc1.weight.detach().cpu().numpy().astype(np.float64).copy()
                self.net.layer1.bias = bp.fc1.bias.detach().cpu().numpy().astype(np.float64).copy()
                self.net.layer0.W = bp.fc2.weight.detach().cpu().numpy().astype(np.float64).copy()
                self.net.layer0.bias = bp.fc2.bias.detach().cpu().numpy().astype(np.float64).copy()
            return
        if init != 'xavier':
            raise ValueError(f"Unknown PC init '{init}'")
        rng = np.random.default_rng(seed)

        def xavier(shape: tuple[int, int]) -> np.ndarray:
            fan_out, fan_in = shape
            bound = np.sqrt(6.0 / float(fan_in + fan_out))
            return rng.uniform(-bound, bound, size=shape)

        self.net.layer1.W = xavier((self.net.layer1.k, self.net.layer1.n))
        self.net.layer1.bias.fill(0.0)
        self.net.layer0.W = xavier((self.net.layer0.k, self.net.layer0.n))
        self.net.layer0.bias.fill(0.0)

    def _run_ticks(self, state: np.ndarray, y_bottom: Optional[np.ndarray],
                   clamp_bottom: bool, n_ticks: int, max_ticks: int) -> None:
        total_ticks = n_ticks if not self.adaptive_inference else max(n_ticks, max_ticks)
        final_delta = float('inf')
        ticks_used = 0
        for ticks_run in range(total_ticks):
            prev_hidden = self.net.layer1.x_state.copy()
            prev_bottom = None if clamp_bottom else self.net.layer0.x_state.copy()
            self.net.tick(state, y_bottom, clamp_top=True, clamp_bottom=clamp_bottom)
            ticks_used = ticks_run + 1
            if not self.adaptive_inference or ticks_run + 1 < n_ticks:
                continue
            max_delta = float(np.max(np.abs(self.net.layer1.x_state - prev_hidden)))
            if prev_bottom is not None:
                max_delta = max(max_delta, float(np.max(np.abs(self.net.layer0.x_state - prev_bottom))))
            final_delta = max_delta
            if max_delta <= self.settle_tol:
                break
        self.settle_calls += 1
        self.settle_ticks += ticks_used
        self.settle_cap_hits += int(self.adaptive_inference and ticks_used >= total_ticks and final_delta > self.settle_tol)
        if final_delta != float('inf'):
            self.settle_final_delta += final_delta

    def diagnostics(self) -> dict[str, float]:
        calls = max(1, self.settle_calls)
        return {
            'pc_value_settle_avg_ticks': self.settle_ticks / calls,
            'pc_value_settle_cap_rate': self.settle_cap_hits / calls,
            'pc_value_settle_avg_delta': self.settle_final_delta / calls,
        }

    def _query_state(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        l0_x = self.net.layer0.x_state.copy()
        l1_x = self.net.layer1.x_state.copy()
        l2_x = self.net.layer2.x_state.copy()
        if target:
            W0_live, b0_live = self.net.layer0.W, self.net.layer0.bias
            W1_live, b1_live = self.net.layer1.W, self.net.layer1.bias
            self.net.layer0.W = self._target_W0
            self.net.layer0.bias = self._target_b0
            self.net.layer1.W = self._target_W1
            self.net.layer1.bias = self._target_b1
        self.net.reset_state()
        self.net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_ticks(np.asarray(state, dtype=np.float64), None, False, self.n_query, self.max_query_ticks)
        out = self.net.x0 * self.value_scale
        if target:
            self.net.layer0.W = W0_live
            self.net.layer0.bias = b0_live
            self.net.layer1.W = W1_live
            self.net.layer1.bias = b1_live
        self.net.layer0.x_state[:] = l0_x
        self.net.layer1.x_state[:] = l1_x
        self.net.layer2.x_state[:] = l2_x
        return np.asarray(out, dtype=np.float64)

    def predict_np(self, states: np.ndarray, target: bool = False) -> np.ndarray:
        arr = np.asarray(states, dtype=np.float64)
        if arr.ndim == 1:
            return self._query_state(arr, target=target)[None, :]
        return np.stack([self._query_state(s, target=target) for s in arr], axis=0)

    def sync_target(self) -> None:
        self._target_W0 = self.net.layer0.W.copy()
        self._target_b0 = self.net.layer0.bias.copy()
        self._target_W1 = self.net.layer1.W.copy()
        self._target_b1 = self.net.layer1.bias.copy()

    def _learn_single(self, state: np.ndarray, target_vec_raw: np.ndarray) -> None:
        target_norm = np.asarray(target_vec_raw, dtype=np.float64) / self.value_scale
        self.net.reset_state()
        self.net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_ticks(np.asarray(state, dtype=np.float64), target_norm, True, self.n_infer, self.max_infer_ticks)
        self.net.set_rates(alpha=self.lr, gamma=0.0)
        for _ in range(self.n_learn):
            self.net.tick(np.asarray(state, dtype=np.float64), target_norm, clamp_top=True, clamp_bottom=True)

    def _pc_gradient_single(self, state: np.ndarray, target_vec_raw: np.ndarray) -> dict[str, np.ndarray]:
        target_norm = np.asarray(target_vec_raw, dtype=np.float64) / self.value_scale
        state64 = np.asarray(state, dtype=np.float64)
        self.net.reset_state()
        self.net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_ticks(state64, target_norm, True, self.n_infer, self.max_infer_ticks)
        hidden_phi, _ = self.net.layer0.phi_fn(self.net.layer1.x_state)
        input_phi, _ = self.net.layer1.phi_fn(self.net.layer2.x_state)
        # PC local increments are +eps*phi. Adam expects dL/dtheta, so negate.
        return {
            'W0': -np.outer(self.net.layer0.eps, hidden_phi),
            'b0': -self.net.layer0.eps.copy(),
            'W1': -np.outer(self.net.layer1.eps, input_phi),
            'b1': -self.net.layer1.eps.copy(),
        }

    def _apply_adam(self, grads: dict[str, np.ndarray]) -> None:
        self._adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = {
            'W0': self.net.layer0.W,
            'b0': self.net.layer0.bias,
            'W1': self.net.layer1.W,
            'b1': self.net.layer1.bias,
        }
        for key, param in params.items():
            grad = grads[key]
            self._adam_m[key] = beta1 * self._adam_m[key] + (1.0 - beta1) * grad
            self._adam_v[key] = beta2 * self._adam_v[key] + (1.0 - beta2) * (grad * grad)
            m_hat = self._adam_m[key] / (1.0 - beta1 ** self._adam_t)
            v_hat = self._adam_v[key] / (1.0 - beta2 ** self._adam_t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            np.clip(param, -self.net.layer0.wclip, self.net.layer0.wclip, out=param)

    def _apply_local_trace(self, grads: dict[str, np.ndarray]) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-6
        params = {
            'W0': self.net.layer0.W,
            'b0': self.net.layer0.bias,
            'W1': self.net.layer1.W,
            'b1': self.net.layer1.bias,
        }
        for key, param in params.items():
            grad = grads[key]
            self._adam_m[key] = beta1 * self._adam_m[key] + (1.0 - beta1) * grad
            self._adam_v[key] = beta2 * self._adam_v[key] + (1.0 - beta2) * (grad * grad)
            param -= self.trace_scale * self.lr * self._adam_m[key] / (np.sqrt(self._adam_v[key]) + eps)
            np.clip(param, -self.net.layer0.wclip, self.net.layer0.wclip, out=param)

    def _learn_batch_adam(self, states: np.ndarray, targets_raw: np.ndarray) -> None:
        grads = {
            'W0': np.zeros_like(self.net.layer0.W),
            'b0': np.zeros_like(self.net.layer0.bias),
            'W1': np.zeros_like(self.net.layer1.W),
            'b1': np.zeros_like(self.net.layer1.bias),
        }
        for state, target_vec in zip(states, targets_raw):
            sample_grads = self._pc_gradient_single(state, target_vec)
            for key in grads:
                grads[key] += sample_grads[key]
        scale = 1.0 / max(1, len(states))
        for key in grads:
            grads[key] *= scale
        self._apply_adam(grads)

    def replay_update(self, memory: list[tuple[np.ndarray, int, float, np.ndarray, bool]],
                      policy_model: 'PolicyModel') -> float:
        if not memory:
            return 0.0
        batch_size = min(BATCH_SIZE, len(memory))
        minibatch = random.sample(memory, batch_size)
        states = []
        targets_raw = []
        losses = []
        for state, action, reward, next_state, done in minibatch:
            target = float(reward)
            if not done:
                next_policy = policy_model.probs_np(np.asarray(next_state, dtype=np.float64))
                next_values = self.predict_np(np.asarray(next_state, dtype=np.float64), target=True)[0]
                target += self.discount * float(np.dot(next_policy[0], next_values))
            current = self.predict_np(np.asarray(state, dtype=np.float64), target=False)[0]
            losses.append(abs(target - current[action]))
            current[action] = target
            states.append(np.asarray(state, dtype=np.float64))
            targets_raw.append(current)
            if self.optimizer == 'hebbian':
                self._learn_single(state, current)
            elif self.optimizer == 'trace':
                self._apply_local_trace(self._pc_gradient_single(state, current))
        if self.optimizer == 'adam' and states:
            self._learn_batch_adam(np.stack(states), np.stack(targets_raw))
        self.last_loss = float(np.mean(losses)) if losses else 0.0
        return self.last_loss

    def distill_from_teacher(self, memory: list[tuple[np.ndarray, int, float, np.ndarray, bool]],
                             teacher_model: BPValueModel) -> float:
        if not memory:
            return 0.0
        batch_size = min(BATCH_SIZE, len(memory))
        minibatch = random.sample(memory, batch_size)
        states = np.stack([item[0] for item in minibatch]).astype(np.float32)
        teacher_targets = teacher_model.predict_np(states)
        losses = []
        for state, target in zip(states, teacher_targets):
            pred = self.predict_np(state, target=False)[0]
            losses.append(float(np.mean((pred - target) ** 2)))
            self._learn_single(state, target)
        self.last_loss = float(np.mean(losses)) if losses else 0.0
        return self.last_loss


PolicyModel = object


class BPPolicyModel:
    def __init__(self, hidden: int, lr: float, seed: int, device: torch.device):
        set_seed(seed)
        self.device = device
        self.net = BPMLP(STATE_SIZE, hidden, ACTION_SIZE).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.last_loss = 0.0

    def logits_np(self, states: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            y = self.net(x).cpu().numpy()
        return y

    def probs_np(self, states: np.ndarray) -> np.ndarray:
        logits = self.logits_np(states)
        if logits.ndim == 1:
            logits = logits[None, :]
        return np.stack([softmax_np(row) for row in logits], axis=0)

    def sample_action(self, state: np.ndarray) -> int:
        x = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        probs = F.softmax(self.net(x), dim=1).squeeze(0)
        return sample_action_from_probs(probs)

    def update_histories(self, histories: list[History], value_model: object) -> float:
        losses = []
        for hist in histories:
            if not hist.states:
                continue
            states = np.asarray(hist.states, dtype=np.float32)
            states_t = torch.as_tensor(states, device=self.device)
            p = F.softmax(self.net(states_t), dim=1)
            with torch.no_grad():
                v = torch.as_tensor(value_model.predict_np(states), dtype=torch.float32, device=self.device)
            losses.append(-(p * F.log_softmax(v, dim=1)).sum(dim=1).mean())
        if not losses:
            self.last_loss = 0.0
            return 0.0
        loss = torch.stack(losses).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss


class PCPolicyModel:
    def __init__(
        self,
        hidden: int,
        lr: float,
        seed: int,
        gamma_pc: float,
        n_infer: int,
        n_learn: int,
        n_query: int,
        adaptive_inference: bool,
        settle_tol: float,
        max_infer_ticks: int,
        max_query_ticks: int,
        policy_clip: float = 5.0,
        hidden_clip: Optional[float] = None,
        smoothing: float = 0.02,
        eps_clip: Optional[float] = None,
        init: str = 'from_bp',
        device: Optional[torch.device] = None,
        optimizer: str = 'adam',
        trace_scale: float = 1.0,
    ):
        set_seed(seed)
        self.lr = lr
        self.gamma_pc = gamma_pc
        self.n_infer = n_infer
        self.n_learn = n_learn
        self.n_query = n_query
        self.adaptive_inference = adaptive_inference
        self.settle_tol = settle_tol
        self.max_infer_ticks = max_infer_ticks
        self.max_query_ticks = max_query_ticks
        self.policy_clip = policy_clip
        self.smoothing = smoothing
        self.optimizer = optimizer
        self.trace_scale = trace_scale
        self.device = device if device is not None else torch.device('cpu')
        self.net = PCNet3Layer(
            k_lut=[ACTION_SIZE, hidden, STATE_SIZE],
            act_lut=['relu', 'linear', 'linear'],
            wclip=20.0,
            xclip_lut=[policy_clip, hidden_clip, None],
            eps_clip_lut=[eps_clip, eps_clip, eps_clip],
            gamma=gamma_pc,
            alpha=lr,
            seed=seed,
            rtl_init=False,
            gen_k_lut=None,
        )
        self._init_weights(init, seed, hidden, device)
        self.last_loss = 0.0
        self.settle_calls = 0
        self.settle_ticks = 0
        self.settle_cap_hits = 0
        self.settle_final_delta = 0.0
        self._adam_t = 0
        self._adam_m = {
            'W0': np.zeros((ACTION_SIZE, hidden), dtype=np.float64),
            'b0': np.zeros(ACTION_SIZE, dtype=np.float64),
            'W1': np.zeros((hidden, STATE_SIZE), dtype=np.float64),
            'b1': np.zeros(hidden, dtype=np.float64),
        }
        self._adam_v = {k: np.zeros_like(v) for k, v in self._adam_m.items()}

    def _init_weights(self, init: str, seed: int, hidden: int, device: Optional[torch.device]) -> None:
        if init == 'pc':
            return
        if init == 'from_bp':
            bp = BPMLP(STATE_SIZE, hidden, ACTION_SIZE)
            if device is not None:
                bp = bp.to(device)
            with torch.no_grad():
                self.net.layer1.W = bp.fc1.weight.detach().cpu().numpy().astype(np.float64).copy()
                self.net.layer1.bias = bp.fc1.bias.detach().cpu().numpy().astype(np.float64).copy()
                self.net.layer0.W = bp.fc2.weight.detach().cpu().numpy().astype(np.float64).copy()
                self.net.layer0.bias = bp.fc2.bias.detach().cpu().numpy().astype(np.float64).copy()
            return
        if init != 'xavier':
            raise ValueError(f"Unknown PC init '{init}'")
        rng = np.random.default_rng(seed)

        def xavier(shape: tuple[int, int]) -> np.ndarray:
            fan_out, fan_in = shape
            bound = np.sqrt(6.0 / float(fan_in + fan_out))
            return rng.uniform(-bound, bound, size=shape)

        self.net.layer1.W = xavier((self.net.layer1.k, self.net.layer1.n))
        self.net.layer1.bias.fill(0.0)
        self.net.layer0.W = xavier((self.net.layer0.k, self.net.layer0.n))
        self.net.layer0.bias.fill(0.0)

    def _run_ticks(self, state: np.ndarray, y_bottom: Optional[np.ndarray],
                   clamp_bottom: bool, n_ticks: int, max_ticks: int) -> None:
        total_ticks = n_ticks if not self.adaptive_inference else max(n_ticks, max_ticks)
        final_delta = float('inf')
        ticks_used = 0
        for ticks_run in range(total_ticks):
            prev_hidden = self.net.layer1.x_state.copy()
            prev_bottom = None if clamp_bottom else self.net.layer0.x_state.copy()
            self.net.tick(state, y_bottom, clamp_top=True, clamp_bottom=clamp_bottom)
            ticks_used = ticks_run + 1
            if not self.adaptive_inference or ticks_run + 1 < n_ticks:
                continue
            max_delta = float(np.max(np.abs(self.net.layer1.x_state - prev_hidden)))
            if prev_bottom is not None:
                max_delta = max(max_delta, float(np.max(np.abs(self.net.layer0.x_state - prev_bottom))))
            final_delta = max_delta
            if max_delta <= self.settle_tol:
                break
        self.settle_calls += 1
        self.settle_ticks += ticks_used
        self.settle_cap_hits += int(self.adaptive_inference and ticks_used >= total_ticks and final_delta > self.settle_tol)
        if final_delta != float('inf'):
            self.settle_final_delta += final_delta

    def diagnostics(self) -> dict[str, float]:
        calls = max(1, self.settle_calls)
        return {
            'pc_policy_settle_avg_ticks': self.settle_ticks / calls,
            'pc_policy_settle_cap_rate': self.settle_cap_hits / calls,
            'pc_policy_settle_avg_delta': self.settle_final_delta / calls,
        }

    def _query_state(self, state: np.ndarray) -> np.ndarray:
        l0_x = self.net.layer0.x_state.copy()
        l1_x = self.net.layer1.x_state.copy()
        l2_x = self.net.layer2.x_state.copy()
        self.net.reset_state()
        self.net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_ticks(np.asarray(state, dtype=np.float64), None, False, self.n_query, self.max_query_ticks)
        logits = self.net.x0.copy()
        self.net.layer0.x_state[:] = l0_x
        self.net.layer1.x_state[:] = l1_x
        self.net.layer2.x_state[:] = l2_x
        return np.asarray(logits, dtype=np.float64)

    def logits_np(self, states: np.ndarray) -> np.ndarray:
        arr = np.asarray(states, dtype=np.float64)
        if arr.ndim == 1:
            return self._query_state(arr)[None, :]
        return np.stack([self._query_state(s) for s in arr], axis=0)

    def probs_np(self, states: np.ndarray) -> np.ndarray:
        logits = self.logits_np(states)
        if logits.ndim == 1:
            logits = logits[None, :]
        return np.stack([softmax_np(row) for row in logits], axis=0)

    def sample_action(self, state: np.ndarray) -> int:
        probs = self.probs_np(np.asarray(state, dtype=np.float64))[0]
        probs_t = torch.as_tensor(probs, dtype=torch.float32, device=self.device)
        return sample_action_from_probs(probs_t)

    def _target_logits_from_values(self, values: np.ndarray) -> np.ndarray:
        greedy = int(np.argmax(values))
        probs = np.full(ACTION_SIZE, self.smoothing / max(1, ACTION_SIZE - 1), dtype=np.float64)
        probs[greedy] = 1.0 - self.smoothing
        logits = np.log(np.clip(probs, 1e-6, 1.0))
        logits -= np.mean(logits)
        return np.clip(logits, -self.policy_clip, self.policy_clip)

    def _learn_single(self, state: np.ndarray, target_logits: np.ndarray) -> None:
        target = np.asarray(target_logits, dtype=np.float64)
        self.net.reset_state()
        self.net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_ticks(np.asarray(state, dtype=np.float64), target, True, self.n_infer, self.max_infer_ticks)
        self.net.set_rates(alpha=self.lr, gamma=0.0)
        for _ in range(self.n_learn):
            self.net.tick(np.asarray(state, dtype=np.float64), target, clamp_top=True, clamp_bottom=True)

    def _policy_loss_grad(self, logits: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
        probs = softmax_np(logits)
        value_log_probs = np.asarray(values, dtype=np.float64)
        value_log_probs = value_log_probs - np.max(value_log_probs)
        value_log_probs = value_log_probs - np.log(np.sum(np.exp(value_log_probs)))
        expected = float(np.dot(probs, value_log_probs))
        loss = -expected
        grad_logits = probs * (expected - value_log_probs)
        return loss, grad_logits

    def _pc_gradient_single(self, state: np.ndarray, values: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        state64 = np.asarray(state, dtype=np.float64)
        logits = self._query_state(state64)
        loss, grad_logits = self._policy_loss_grad(logits, values)
        # A bottom target of z - dL/dz makes eps = -dL/dz, so PC local
        # increments encode the negative gradient.
        target = logits - grad_logits
        self.net.reset_state()
        self.net.set_rates(alpha=0.0, gamma=self.gamma_pc)
        self._run_ticks(state64, target, True, self.n_infer, self.max_infer_ticks)
        hidden_phi, _ = self.net.layer0.phi_fn(self.net.layer1.x_state)
        input_phi, _ = self.net.layer1.phi_fn(self.net.layer2.x_state)
        return loss, {
            'W0': -np.outer(self.net.layer0.eps, hidden_phi),
            'b0': -self.net.layer0.eps.copy(),
            'W1': -np.outer(self.net.layer1.eps, input_phi),
            'b1': -self.net.layer1.eps.copy(),
        }

    def _apply_adam(self, grads: dict[str, np.ndarray]) -> None:
        self._adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = {
            'W0': self.net.layer0.W,
            'b0': self.net.layer0.bias,
            'W1': self.net.layer1.W,
            'b1': self.net.layer1.bias,
        }
        for key, param in params.items():
            grad = grads[key]
            self._adam_m[key] = beta1 * self._adam_m[key] + (1.0 - beta1) * grad
            self._adam_v[key] = beta2 * self._adam_v[key] + (1.0 - beta2) * (grad * grad)
            m_hat = self._adam_m[key] / (1.0 - beta1 ** self._adam_t)
            v_hat = self._adam_v[key] / (1.0 - beta2 ** self._adam_t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            np.clip(param, -self.net.layer0.wclip, self.net.layer0.wclip, out=param)

    def _apply_local_trace(self, grads: dict[str, np.ndarray]) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-6
        params = {
            'W0': self.net.layer0.W,
            'b0': self.net.layer0.bias,
            'W1': self.net.layer1.W,
            'b1': self.net.layer1.bias,
        }
        for key, param in params.items():
            grad = grads[key]
            self._adam_m[key] = beta1 * self._adam_m[key] + (1.0 - beta1) * grad
            self._adam_v[key] = beta2 * self._adam_v[key] + (1.0 - beta2) * (grad * grad)
            param -= self.trace_scale * self.lr * self._adam_m[key] / (np.sqrt(self._adam_v[key]) + eps)
            np.clip(param, -self.net.layer0.wclip, self.net.layer0.wclip, out=param)

    def _update_histories_adam(self, histories: list[History], value_model: object) -> float:
        grads = {
            'W0': np.zeros_like(self.net.layer0.W),
            'b0': np.zeros_like(self.net.layer0.bias),
            'W1': np.zeros_like(self.net.layer1.W),
            'b1': np.zeros_like(self.net.layer1.bias),
        }
        losses = []
        n = 0
        for hist in histories:
            if not hist.states:
                continue
            states = np.asarray(hist.states, dtype=np.float64)
            values_batch = value_model.predict_np(states)
            for state, values in zip(states, values_batch):
                loss, sample_grads = self._pc_gradient_single(state, values)
                losses.append(loss)
                n += 1
                for key in grads:
                    grads[key] += sample_grads[key]
        if n:
            scale = 1.0 / n
            for key in grads:
                grads[key] *= scale
            self._apply_adam(grads)
        return float(np.mean(losses)) if losses else 0.0

    def _update_histories_trace(self, histories: list[History], value_model: object) -> float:
        losses = []
        for hist in histories:
            if not hist.states:
                continue
            states = np.asarray(hist.states, dtype=np.float64)
            values_batch = value_model.predict_np(states)
            for state, values in zip(states, values_batch):
                loss, sample_grads = self._pc_gradient_single(state, values)
                losses.append(loss)
                self._apply_local_trace(sample_grads)
        return float(np.mean(losses)) if losses else 0.0

    def update_histories(self, histories: list[History], value_model: object) -> float:
        if self.optimizer == 'adam':
            self.last_loss = self._update_histories_adam(histories, value_model)
            return self.last_loss
        if self.optimizer == 'trace':
            self.last_loss = self._update_histories_trace(histories, value_model)
            return self.last_loss
        losses = []
        for hist in histories:
            for state in hist.states:
                values = value_model.predict_np(np.asarray(state, dtype=np.float64))[0]
                target_logits = self._target_logits_from_values(values)
                pred = self.logits_np(np.asarray(state, dtype=np.float64))[0]
                losses.append(float(np.mean((pred - target_logits) ** 2)))
                self._learn_single(state, target_logits)
        self.last_loss = float(np.mean(losses)) if losses else 0.0
        return self.last_loss


def make_models(args: argparse.Namespace, device: torch.device):
    if args.value_backend == 'bp':
        value_model = BPValueModel(args.hidden, args.lr_value, args.discount, args.seed, device)
    else:
        value_model = PCValueModel(
            hidden=args.hidden,
            lr=args.lr_value,
            discount=args.discount,
            seed=args.seed,
            gamma_pc=args.gamma_pc,
            n_infer=args.pc_infer,
            n_learn=args.pc_learn,
            n_query=args.pc_query,
            adaptive_inference=not args.no_adaptive_inference,
            settle_tol=args.settle_tol,
            max_infer_ticks=args.max_infer_ticks,
            max_query_ticks=args.max_query_ticks,
            value_scale=args.pc_value_scale,
            value_clip=args.pc_value_clip,
            hidden_clip=args.pc_hidden_clip,
            eps_clip=args.pc_eps_clip,
            init=args.pc_init,
            device=device,
            optimizer=args.pc_optimizer,
            trace_scale=args.pc_trace_scale,
        )
    if args.policy_backend == 'bp':
        policy_model = BPPolicyModel(args.hidden, args.lr_policy, args.seed + 1, device)
    else:
        policy_model = PCPolicyModel(
            hidden=args.hidden,
            lr=args.lr_policy,
            seed=args.seed + 1,
            gamma_pc=args.gamma_pc,
            n_infer=args.pc_infer,
            n_learn=args.pc_learn,
            n_query=args.pc_query,
            adaptive_inference=not args.no_adaptive_inference,
            settle_tol=args.settle_tol,
            max_infer_ticks=args.max_infer_ticks,
            max_query_ticks=args.max_query_ticks,
            policy_clip=args.pc_policy_clip,
            hidden_clip=args.pc_hidden_clip,
            smoothing=args.pc_policy_smoothing,
            eps_clip=args.pc_eps_clip,
            init=args.pc_init,
            device=device,
            optimizer=args.pc_optimizer,
            trace_scale=args.pc_trace_scale,
        )
    return value_model, policy_model


def main(args: argparse.Namespace) -> tuple[list[int], list[float], list[float]]:
    set_seed(args.seed)
    device = torch.device('cpu')
    env = gym.make('CartPole-v1')
    env.reset(seed=args.seed)
    value_model, policy_model = make_models(args, device)
    teacher_value_model = None
    if args.value_backend == 'pc' and args.distill_value_from_bp_teacher:
        teacher_value_model = BPValueModel(args.hidden, args.lr_value, args.discount, args.seed + 99, device)

    avgreward = 0.0
    histories: list[History] = []
    rewards: list[int] = []
    plosses: list[float] = []
    vlosses: list[float] = []
    memory: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'avg_reward_ema', 'policy_loss', 'value_loss',
                         'value_backend', 'policy_backend',
                         'pc_value_settle_avg_ticks', 'pc_value_settle_cap_rate',
                         'pc_value_settle_avg_delta',
                         'pc_policy_settle_avg_ticks', 'pc_policy_settle_cap_rate',
                         'pc_policy_settle_avg_delta'])

        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            state = np.asarray(obs, dtype=np.float32)
            history = History(STATE_SIZE, ACTION_SIZE, args.discount)
            episode_reward = 0

            for _ in range(10000):
                action = policy_model.sample_action(state)
                obs, reward, terminated, truncated, _ = env.step(action)
                next_state = np.asarray(obs, dtype=np.float32)
                done = bool(terminated or truncated)

                history.states.append(state.copy())
                history.actions.append(action)
                history.rewards.append(float(reward))
                if len(memory) == MEM_SIZE:
                    del memory[0]
                memory.append((state.copy(), action, float(reward), next_state.copy(), done))

                state = next_state
                episode_reward += reward
                if done:
                    break

            histories.append(history)
            avgreward = 0.1 * episode_reward + 0.9 * avgreward

            if episode % args.infotime == 0:
                print(f'(episode:{episode}, avgreward:{avgreward})')

            if episode % args.policy_train_every == 0:
                ploss = policy_model.update_histories(histories, value_model)
                histories = []
            else:
                ploss = policy_model.last_loss

            if episode % args.target_update_every == 0:
                value_model.sync_target()
                if teacher_value_model is not None:
                    teacher_value_model.sync_target()

            replay_losses = []
            for _ in range(args.replay_updates_per_episode):
                if teacher_value_model is not None:
                    teacher_value_model.replay_update(memory, policy_model)
                    replay_losses.append(value_model.distill_from_teacher(memory, teacher_value_model))
                else:
                    replay_losses.append(value_model.replay_update(memory, policy_model))
            vloss = float(np.mean(replay_losses)) if replay_losses else 0.0

            rewards.append(int(episode_reward))
            plosses.append(float(ploss))
            vlosses.append(float(vloss))
            diag = value_model.diagnostics() if hasattr(value_model, 'diagnostics') else {}
            if hasattr(policy_model, 'diagnostics'):
                diag.update(policy_model.diagnostics())
            writer.writerow([episode, episode_reward, avgreward, plosses[-1], vlosses[-1],
                             args.value_backend, args.policy_backend,
                             diag.get('pc_value_settle_avg_ticks', ''),
                             diag.get('pc_value_settle_cap_rate', ''),
                             diag.get('pc_value_settle_avg_delta', ''),
                             diag.get('pc_policy_settle_avg_ticks', ''),
                             diag.get('pc_policy_settle_cap_rate', ''),
                             diag.get('pc_policy_settle_avg_delta', '')])

    env.close()
    return rewards, plosses, vlosses


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=2000)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--discount', type=float, default=0.99)
    ap.add_argument('--lr-policy', type=float, default=0.001)
    ap.add_argument('--lr-value', type=float, default=0.001)
    ap.add_argument('--infotime', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-csv', type=str, default='python_runs/cartpole_millidge_hybrid.csv')
    ap.add_argument('--value-backend', choices=['bp', 'pc'], default='bp')
    ap.add_argument('--policy-backend', choices=['bp', 'pc'], default='bp')
    ap.add_argument('--policy-train-every', type=int, default=5)
    ap.add_argument('--target-update-every', type=int, default=50)
    ap.add_argument('--replay-updates-per-episode', type=int, default=1)
    ap.add_argument('--distill-value-from-bp-teacher', action='store_true')
    ap.add_argument('--gamma-pc', type=float, default=0.1)
    ap.add_argument('--pc-infer', type=int, default=50)
    ap.add_argument('--pc-learn', type=int, default=10)
    ap.add_argument('--pc-query', type=int, default=100)
    ap.add_argument('--settle-tol', type=float, default=0.001)
    ap.add_argument('--max-infer-ticks', type=int, default=200)
    ap.add_argument('--max-query-ticks', type=int, default=300)
    ap.add_argument('--no-adaptive-inference', action='store_true')
    ap.add_argument('--pc-value-scale', type=float, default=1.0)
    ap.add_argument('--pc-value-clip', type=float, default=None)
    ap.add_argument('--pc-hidden-clip', type=float, default=None)
    ap.add_argument('--pc-policy-clip', type=float, default=5.0)
    ap.add_argument('--pc-policy-smoothing', type=float, default=0.02)
    ap.add_argument('--pc-eps-clip', type=float, default=None)
    ap.add_argument('--pc-init', choices=['from_bp', 'xavier', 'pc'], default='from_bp')
    ap.add_argument('--pc-optimizer', choices=['adam', 'trace', 'hebbian'], default='adam')
    ap.add_argument('--pc-trace-scale', type=float, default=1.0)
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rewards, plosses, vlosses = main(args)
    print({
        'value_backend': args.value_backend,
        'policy_backend': args.policy_backend,
        'final_avg50': float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards)),
        'best': int(max(rewards)) if rewards else 0,
        'episodes': len(rewards),
        'csv': args.out_csv,
    })
