#!/usr/bin/env python3
"""TD3 / BP Deep-AIF baseline for MuJoCo continuous control.

The `q` critic semantics are conventional TD3. The `aif` critic semantics are
the Millidge-style amortized expected-free-energy view with G(s, a) = -Q(s, a):
critics learn G, targets use the TD3 min-Q rule as a max-G rule, and the actor
minimizes G instead of maximizing Q.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is false")
    return torch.device(name)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, action_scale: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, act_dim)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.action_scale * torch.tanh(self.out(x))


class PCActor(Actor):
    """Exact-local deterministic actor update for TD3/AIF."""

    def exactlocal_update(
        self,
        args: argparse.Namespace,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        states: torch.Tensor,
    ) -> float:
        z1 = self.fc1(states)
        h1 = F.relu(z1)
        z2 = self.fc2(h1)
        h2 = F.relu(z2)
        z3 = self.out(h2)
        actions = self.action_scale * torch.tanh(z3)

        action_probe = actions.detach().requires_grad_(True)
        actor_loss = actor_objective(args, critic, states, action_probe)
        (grad_actions,) = torch.autograd.grad(actor_loss, action_probe)

        optimizer.zero_grad()
        with torch.no_grad():
            batch = states.shape[0]
            dz3 = grad_actions * self.action_scale * (1.0 - torch.tanh(z3).square())
            dz2 = (dz3 @ self.out.weight) * (z2 > 0.0).float()
            dz1 = (dz2 @ self.fc2.weight) * (z1 > 0.0).float()

            self.out.weight.grad = ((dz3.T @ h2) / batch).detach().clone()
            self.out.bias.grad = dz3.mean(dim=0).detach().clone()
            self.fc2.weight.grad = ((dz2.T @ h1) / batch).detach().clone()
            self.fc2.bias.grad = dz2.mean(dim=0).detach().clone()
            self.fc1.weight.grad = ((dz1.T @ states) / batch).detach().clone()
            self.fc1.bias.grad = dz1.mean(dim=0).detach().clone()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(actor_loss.detach().cpu().item())


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class AmortizedEFECritic(Critic):
    """BP amortized EFE estimator G(s, a) for the Deep-AIF TD3 bridge."""

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return -super().forward(states, actions)


class PCTickCritic(nn.Module):
    """Two-hidden-layer exact-local PC critic for TD3/AIF.

    Matches the BP Critic architecture (fc1 → relu → fc2 → relu → out).
    In AIF mode the public forward returns G = -Q_raw.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int,
        gamma_pc: float,
        query_ticks: int,
        value_scale: float,
        signed_efe: bool,
    ):
        super().__init__()
        self.hidden = hidden
        self.gamma_pc = gamma_pc
        self.query_ticks = query_ticks
        self.value_scale = value_scale
        self.signed_efe = signed_efe
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)

    def _input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([states, actions], dim=1)

    def forward_raw_norm(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(self._input(states, actions)))
        return self.out(F.relu(self.fc2(x)))

    def forward_raw(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward_raw_norm(states, actions) * self.value_scale

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raw = self.forward_raw(states, actions)
        return -raw if self.signed_efe else raw

    def tick_query_raw_norm(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Settle hidden states via PC dynamics; return (pred_norm, h2_state, h1_state)."""
        inputs = self._input(states, actions)
        batch = inputs.shape[0]
        dtype, device = inputs.dtype, inputs.device
        h1 = torch.full((batch, self.hidden), 0.001, dtype=dtype, device=device)
        h2 = torch.full((batch, self.hidden), 0.001, dtype=dtype, device=device)
        x0 = torch.full((batch, 1), 0.001, dtype=dtype, device=device)
        back_h2 = torch.zeros((batch, self.hidden), dtype=dtype, device=device)
        back_h1 = torch.zeros((batch, self.hidden), dtype=dtype, device=device)
        for _ in range(self.query_ticks):
            mu_h1 = F.linear(inputs, self.fc1.weight, self.fc1.bias)
            eps_h1 = h1 - mu_h1
            h1 = h1 + self.gamma_pc * (back_h1 - eps_h1)
            phi_h1 = F.relu(h1)

            mu_h2 = F.linear(phi_h1, self.fc2.weight, self.fc2.bias)
            eps_h2 = h2 - mu_h2
            h2 = h2 + self.gamma_pc * (back_h2 - eps_h2)
            phi_h2 = F.relu(h2)

            mu_out = F.linear(phi_h2, self.out.weight, self.out.bias)
            eps_out = x0 - mu_out
            back_h2 = eps_out @ self.out.weight
            x0 = x0 - self.gamma_pc * eps_out

            back_h1 = eps_h2 @ self.fc2.weight
        return x0, h2, h1

    def exactlocal_update(
        self,
        optimizer: torch.optim.Optimizer,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        optimizer.zero_grad()
        with torch.no_grad():
            inputs = self._input(states, actions)
            pred_norm, h2, h1 = self.tick_query_raw_norm(states, actions)
            target_raw = -targets if self.signed_efe else targets
            target_norm = target_raw / self.value_scale
            out_delta = pred_norm - target_norm
            phi_h2 = F.relu(h2)
            phi_h1 = F.relu(h1)
            d_h2 = (out_delta @ self.out.weight) * (h2 > 0.0).float()
            d_h1 = (d_h2 @ self.fc2.weight) * (h1 > 0.0).float()
            batch = states.shape[0]
            loss = F.mse_loss(
                (-pred_norm if self.signed_efe else pred_norm) * self.value_scale,
                targets,
            )
            self.out.weight.grad = ((out_delta.T @ phi_h2) / batch).detach().clone()
            self.out.bias.grad = out_delta.mean(dim=0).detach().clone()
            self.fc2.weight.grad = ((d_h2.T @ phi_h1) / batch).detach().clone()
            self.fc2.bias.grad = d_h2.mean(dim=0).detach().clone()
            self.fc1.weight.grad = ((d_h1.T @ inputs) / batch).detach().clone()
            self.fc1.bias.grad = d_h1.mean(dim=0).detach().clone()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach().cpu().item())

    def bp_equiv_update(
        self,
        optimizer: torch.optim.Optimizer,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        grad_clip: float,
    ) -> float:
        """Update the PC critic with the exact BP gradient for this substrate."""
        pred = self.forward(states, actions)
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach().cpu().item())

    def measure_bp_cosine(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, float]:
        """Return (grad_cosine_vs_bp, tick_vs_direct_mse) for diagnostic."""
        inputs = self._input(states, actions)
        target_norm = ((-targets if self.signed_efe else targets) / self.value_scale)

        # BP gradient via autograd on direct forward
        pred_direct = self.forward_raw_norm(states, actions)
        bp_loss = F.mse_loss(pred_direct, target_norm)

        # Temporarily enable grads for BP computation
        for p in self.parameters():
            p.requires_grad_(True)
        self.zero_grad()
        pred_direct2 = self.forward_raw_norm(states, actions)
        F.mse_loss(pred_direct2, target_norm.detach()).backward()
        bp_grads = torch.cat([
            self.fc1.weight.grad.flatten(),
            self.fc1.bias.grad.flatten(),
            self.fc2.weight.grad.flatten(),
            self.fc2.bias.grad.flatten(),
            self.out.weight.grad.flatten(),
            self.out.bias.grad.flatten(),
        ]).clone()
        self.zero_grad()
        for p in self.parameters():
            p.requires_grad_(True)

        # PC exact-local gradient
        with torch.no_grad():
            pred_tick, h2, h1 = self.tick_query_raw_norm(states, actions)
            phi_h2 = F.relu(h2)
            phi_h1 = F.relu(h1)
            out_delta = pred_tick - target_norm
            d_h2 = (out_delta @ self.out.weight) * (h2 > 0.0).float()
            d_h1 = (d_h2 @ self.fc2.weight) * (h1 > 0.0).float()
            batch = states.shape[0]
            pc_grads = torch.cat([
                (d_h1.T @ inputs / batch).flatten(),
                d_h1.mean(dim=0).flatten(),
                (d_h2.T @ phi_h1 / batch).flatten(),
                d_h2.mean(dim=0).flatten(),
                (out_delta.T @ phi_h2 / batch).flatten(),
                out_delta.mean(dim=0).flatten(),
            ])

            cosine = float(F.cosine_similarity(bp_grads.unsqueeze(0), pc_grads.unsqueeze(0)).item())
            tick_mse = float(F.mse_loss(pred_tick, pred_direct.detach()).item())
        return cosine, tick_mse


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: Deque[Transition] = deque(maxlen=capacity)

    def append(self, item: Transition) -> None:
        self.data.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.data, batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.stack([t.action for t in batch]), dtype=torch.float32, device=device)
        rewards = torch.as_tensor([[t.reward] for t in batch], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
        dones = torch.as_tensor([[t.done] for t in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def make_env(env_name: str, seed: int | None = None) -> gym.Env:
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass
    return env


def evaluate(actor: nn.Module, env_name: str, seed: int, device: torch.device, episodes: int) -> float:
    env = make_env(env_name)
    rewards = []
    for idx in range(episodes):
        obs, _ = env.reset(seed=seed + 10000 + idx)
        done = False
        total = 0.0
        while not done:
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_t).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            total += float(reward)
            done = bool(terminated or truncated)
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def td3_target(
    args: argparse.Namespace,
    critic1_target: nn.Module,
    critic2_target: nn.Module,
    next_states: torch.Tensor,
    next_actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    target1 = critic1_target(next_states, next_actions)
    target2 = critic2_target(next_states, next_actions)
    if args.critic_semantics == "q":
        target_value = torch.minimum(target1, target2)
        return rewards + args.discount * (1.0 - dones) * target_value
    # G = -Q, so min(Q1, Q2) = -max(G1, G2), and target_G = -target_Q.
    target_g = torch.maximum(target1, target2)
    return -rewards + args.discount * (1.0 - dones) * target_g


def actor_objective(args: argparse.Namespace, critic: nn.Module, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    value = critic(states, actions)
    if args.critic_semantics == "q":
        return -value.mean()
    return value.mean()


def run(args: argparse.Namespace) -> list[float]:
    set_seed(args.seed)
    device = select_device(args.device)
    env = make_env(args.env, args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    if not np.allclose(action_low, -action_high):
        raise ValueError("This runner expects symmetric continuous action bounds")
    action_scale = action_high
    action_scale_t = torch.as_tensor(action_scale, dtype=torch.float32, device=device).unsqueeze(0)
    action_noise_scale_t = action_scale_t if args.scale_action_noise else torch.ones_like(action_scale_t)
    action_noise_scale_np = action_scale if args.scale_action_noise else np.ones_like(action_scale)

    if args.print_device:
        print({
            "device": str(device),
            "env": args.env,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "action_scale": action_scale.tolist(),
        }, flush=True)

    actor_cls = PCActor if args.actor_backend == "pc" else Actor
    actor = actor_cls(obs_dim, act_dim, args.hidden, action_scale).to(device)
    actor_target = actor_cls(obs_dim, act_dim, args.hidden, action_scale).to(device)
    if args.critic_backend == "pc":
        critic1 = PCTickCritic(
            obs_dim, act_dim, args.hidden, args.gamma_pc, args.pc_query,
            args.pc_critic_value_scale, args.critic_semantics == "aif",
        ).to(device)
        critic2 = PCTickCritic(
            obs_dim, act_dim, args.hidden, args.gamma_pc, args.pc_query,
            args.pc_critic_value_scale, args.critic_semantics == "aif",
        ).to(device)
        critic1_target = PCTickCritic(
            obs_dim, act_dim, args.hidden, args.gamma_pc, args.pc_query,
            args.pc_critic_value_scale, args.critic_semantics == "aif",
        ).to(device)
        critic2_target = PCTickCritic(
            obs_dim, act_dim, args.hidden, args.gamma_pc, args.pc_query,
            args.pc_critic_value_scale, args.critic_semantics == "aif",
        ).to(device)
    else:
        critic_cls = AmortizedEFECritic if args.critic_semantics == "aif" else Critic
        critic1 = critic_cls(obs_dim, act_dim, args.hidden).to(device)
        critic2 = critic_cls(obs_dim, act_dim, args.hidden).to(device)
        critic1_target = critic_cls(obs_dim, act_dim, args.hidden).to(device)
        critic2_target = critic_cls(obs_dim, act_dim, args.hidden).to(device)
    hard_update(actor_target, actor)
    hard_update(critic1_target, critic1)
    hard_update(critic2_target, critic2)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=args.lr_critic)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=args.lr_critic)
    replay = ReplayBuffer(args.replay_size)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rewards: list[float] = []
    total_steps = 0
    update_steps = 0
    best_eval = -float("inf")
    best_eval_episode = 0
    best_actor_state: dict[str, torch.Tensor] | None = None
    actor_frozen = False

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "avg10", "eval_reward", "best_eval",
            "critic_loss", "actor_loss", "steps", "replay_size",
            "grad_cosine", "tick_vs_direct_mse",
        ])
        episode = 0
        next_eval_step = args.eval_every_steps if args.eval_every_steps > 0 else None
        while (
            (args.total_timesteps > 0 and total_steps < args.total_timesteps)
            or (args.total_timesteps <= 0 and episode < args.episodes)
        ):
            episode += 1
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            ep_critic_losses = []
            ep_actor_losses = []

            while not done:
                if total_steps < args.start_steps:
                    action = env.action_space.sample()
                else:
                    state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = actor(state_t).squeeze(0).cpu().numpy()
                    action = action + np.random.normal(0.0, args.exploration_noise, size=act_dim) * action_noise_scale_np
                    action = np.clip(action, action_low, action_high)

                next_obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
                done = bool(terminated or truncated)
                replay.append(Transition(
                    state=np.asarray(obs, dtype=np.float32),
                    action=np.asarray(action, dtype=np.float32),
                    reward=args.reward_scale * float(reward),
                    next_state=np.asarray(next_obs, dtype=np.float32),
                    done=done,
                ))
                obs = next_obs
                episode_reward += float(reward)
                total_steps += 1

                if len(replay) >= args.batch_size and total_steps >= args.update_after:
                    for _ in range(args.updates_per_step):
                        update_steps += 1
                        states, actions, batch_rewards, next_states, dones = replay.sample(args.batch_size, device)
                        with torch.no_grad():
                            noise = torch.randn_like(actions) * args.policy_noise * action_noise_scale_t
                            noise = noise.clamp(-args.noise_clip, args.noise_clip)
                            next_actions = (actor_target(next_states) + noise).clamp(
                                -action_scale_t,
                                action_scale_t,
                            )
                            y = td3_target(
                                args,
                                critic1_target,
                                critic2_target,
                                next_states,
                                next_actions,
                                batch_rewards,
                                dones,
                        )

                        if args.critic_backend == "pc":
                            if args.pc_critic_gradient_mode == "bp_equiv":
                                critic1_loss_value = critic1.bp_equiv_update(
                                    critic1_opt, states, actions, y, args.grad_clip,
                                )
                                critic2_loss_value = critic2.bp_equiv_update(
                                    critic2_opt, states, actions, y, args.grad_clip,
                                )
                            else:
                                critic1_loss_value = critic1.exactlocal_update(critic1_opt, states, actions, y)
                                critic2_loss_value = critic2.exactlocal_update(critic2_opt, states, actions, y)
                            ep_critic_losses.append(critic1_loss_value + critic2_loss_value)
                        else:
                            q1 = critic1(states, actions)
                            q2 = critic2(states, actions)
                            critic1_loss = F.mse_loss(q1, y)
                            critic2_loss = F.mse_loss(q2, y)
                            critic1_opt.zero_grad()
                            critic1_loss.backward()
                            if args.grad_clip > 0:
                                nn.utils.clip_grad_norm_(critic1.parameters(), args.grad_clip)
                            critic1_opt.step()
                            critic2_opt.zero_grad()
                            critic2_loss.backward()
                            if args.grad_clip > 0:
                                nn.utils.clip_grad_norm_(critic2.parameters(), args.grad_clip)
                            critic2_opt.step()
                            ep_critic_losses.append(float((critic1_loss + critic2_loss).detach().cpu().item()))

                        if update_steps % args.policy_delay == 0:
                            if not actor_frozen:
                                if args.actor_backend == "pc":
                                    actor_loss_value = actor.exactlocal_update(args, critic1, actor_opt, states)
                                else:
                                    actor_loss = actor_objective(args, critic1, states, actor(states))
                                    actor_opt.zero_grad()
                                    actor_loss.backward()
                                    if args.grad_clip > 0:
                                        nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                                    actor_opt.step()
                                    actor_loss_value = float(actor_loss.detach().cpu().item())
                                soft_update(actor_target, actor, args.tau)
                                ep_actor_losses.append(actor_loss_value)
                            soft_update(critic1_target, critic1, args.tau)
                            soft_update(critic2_target, critic2, args.tau)

            rewards.append(episode_reward)
            avg10 = float(np.mean(rewards[-10:]))
            eval_reward = ""
            should_eval_episode = args.eval_every > 0 and episode % args.eval_every == 0
            should_eval_step = next_eval_step is not None and total_steps >= next_eval_step
            if should_eval_episode or should_eval_step:
                eval_marker_step = total_steps
                eval_reward = evaluate(actor, args.env, args.seed + episode, device, args.eval_episodes)
                if float(eval_reward) > best_eval:
                    best_eval = float(eval_reward)
                    best_eval_episode = episode
                    best_actor_state = copy.deepcopy(actor.state_dict())
                if args.freeze_actor_after_eval > 0 and not actor_frozen and float(eval_reward) >= args.freeze_actor_after_eval:
                    actor_frozen = True
                    print({"actor_frozen": True, "episode": episode, "eval_reward": float(eval_reward)}, flush=True)
                if next_eval_step is not None:
                    while next_eval_step <= eval_marker_step:
                        next_eval_step += args.eval_every_steps

            critic_loss = float(np.mean(ep_critic_losses)) if ep_critic_losses else 0.0
            actor_loss = float(np.mean(ep_actor_losses)) if ep_actor_losses else 0.0

            grad_cosine = ""
            tick_vs_direct_mse = ""
            if (
                args.critic_drift_every > 0
                and episode % args.critic_drift_every == 0
                and args.critic_backend == "pc"
                and len(replay) >= args.batch_size
            ):
                d_states, d_actions, d_rewards, d_next_states, d_dones = replay.sample(args.batch_size, device)
                with torch.no_grad():
                    d_noise = torch.randn_like(d_actions) * args.policy_noise * action_scale_t
                    d_noise = d_noise.clamp(-args.noise_clip, args.noise_clip)
                    d_next_actions = (actor_target(d_next_states) + d_noise).clamp(-action_scale_t, action_scale_t)
                    d_y = td3_target(args, critic1_target, critic2_target, d_next_states, d_next_actions, d_rewards, d_dones)
                gc, tmse = critic1.measure_bp_cosine(d_states, d_actions, d_y)
                grad_cosine = gc
                tick_vs_direct_mse = tmse
                if episode % args.infotime == 0:
                    print({"episode": episode, "grad_cosine": gc, "tick_vs_direct_mse": tmse}, flush=True)

            if episode % args.infotime == 0:
                print({
                    "episode": episode,
                    "reward": episode_reward,
                    "avg10": avg10,
                    "eval_reward": eval_reward,
                    "best_eval": best_eval if best_eval > -float("inf") else "",
                    "steps": total_steps,
                }, flush=True)
            writer.writerow([
                episode, episode_reward, avg10, eval_reward,
                best_eval if best_eval > -float("inf") else "",
                critic_loss, actor_loss, total_steps, len(replay),
                grad_cosine, tick_vs_direct_mse,
            ])

    env.close()
    final_eval = ""
    best_checkpoint_eval = ""
    if args.final_eval_episodes > 0:
        final_eval = evaluate(actor, args.env, args.seed + 50000, device, args.final_eval_episodes)
        if best_actor_state is not None:
            actor.load_state_dict(best_actor_state)
            best_checkpoint_eval = evaluate(actor, args.env, args.seed + 60000, device, args.final_eval_episodes)
    if best_eval_episode:
        print({"best_eval": best_eval, "best_eval_episode": best_eval_episode}, flush=True)
    if args.final_eval_episodes > 0:
        print({
            "final_policy_eval": final_eval,
            "best_checkpoint_eval": best_checkpoint_eval,
            "final_eval_episodes": args.final_eval_episodes,
        }, flush=True)
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="InvertedPendulum-v5")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--total-timesteps", type=int, default=0,
                        help="If > 0, train until this many environment steps instead of using --episodes")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--actor-backend", choices=["bp", "pc"], default="bp")
    parser.add_argument("--critic-backend", choices=["bp", "pc"], default="bp")
    parser.add_argument("--critic-semantics", choices=["q", "aif"], default="q")
    parser.add_argument("--gamma-pc", type=float, default=0.2)
    parser.add_argument("--pc-query", type=int, default=100)
    parser.add_argument("--pc-critic-value-scale", type=float, default=100.0)
    parser.add_argument("--pc-critic-gradient-mode", choices=["exactlocal", "bp_equiv"], default="exactlocal")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1000000)
    parser.add_argument("--start-steps", type=int, default=25000)
    parser.add_argument("--update-after", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--scale-action-noise", action="store_true",
                        help="Use legacy action-range-relative noise instead of SB3-style absolute action-unit noise")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=0,
                        help="Evaluate every N environment steps, matching SB3-style reporting")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=10)
    parser.add_argument("--freeze-actor-after-eval", type=float, default=0.0,
                        help="Freeze actor updates once eval reaches this threshold (0 = disabled)")
    parser.add_argument("--critic-drift-every", type=int, default=0,
                        help="Measure PC critic grad cosine vs BP every N episodes (0 = disabled)")
    parser.add_argument("--infotime", type=int, default=10)
    parser.add_argument("--out-csv", default="python_runs/mujoco_td3.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--print-device", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    rewards = run(parsed)
    print({
        "episodes": len(rewards),
        "target_episodes": parsed.episodes,
        "target_timesteps": parsed.total_timesteps,
        "final_avg10": float(np.mean(rewards[-10:])) if rewards else 0.0,
        "best": float(np.max(rewards)) if rewards else 0.0,
        "csv": parsed.out_csv,
    })
