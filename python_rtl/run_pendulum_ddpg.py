#!/usr/bin/env python3
"""Pendulum-v1 continuous-control baseline.

This is the Pendulum stepping stone after the CartPole PC/PC reproduction.
It starts with a small CUDA DDPG-style actor/critic so we have a reliable
continuous-control baseline before swapping in PC critic/policy modules.
"""
from __future__ import annotations

import argparse
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


STATE_SIZE = 3
ACTION_SIZE = 1
ACTION_LIMIT = 2.0


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
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, ACTION_SIZE)
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
        return ACTION_LIMIT * torch.tanh(self.out(x))


class PCActor(Actor):
    """PC-shaped deterministic policy for continuous actions.

    Fast mode uses the normal autograd bridge. Exact-local mode asks the critic
    for the action-gradient, then applies derivative-gated local layer updates.
    """

    def exactlocal_update(
        self,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        states: torch.Tensor,
        grad_clip: float,
    ) -> float:
        z1 = self.fc1(states)
        h1 = F.relu(z1)
        z2 = self.fc2(h1)
        h2 = F.relu(z2)
        z3 = self.out(h2)
        actions = ACTION_LIMIT * torch.tanh(z3)

        action_probe = actions.detach().requires_grad_(True)
        actor_loss = -critic(states, action_probe).mean()
        (grad_actions,) = torch.autograd.grad(actor_loss, action_probe)

        with torch.no_grad():
            batch = states.shape[0]
            dz3 = grad_actions * ACTION_LIMIT * (1.0 - torch.tanh(z3).square())
            dz2 = (dz3 @ self.out.weight) * (z2 > 0.0).float()
            dz1 = (dz2 @ self.fc2.weight) * (z1 > 0.0).float()

            grad_out_w = (dz3.T @ h2) / batch
            grad_out_b = dz3.mean(dim=0)
            grad_fc2_w = (dz2.T @ h1) / batch
            grad_fc2_b = dz2.mean(dim=0)
            grad_fc1_w = (dz1.T @ states) / batch
            grad_fc1_b = dz1.mean(dim=0)

        optimizer.zero_grad()
        self.out.weight.grad = grad_out_w.detach().clone()
        self.out.bias.grad = grad_out_b.detach().clone()
        self.fc2.weight.grad = grad_fc2_w.detach().clone()
        self.fc2.bias.grad = grad_fc2_b.detach().clone()
        self.fc1.weight.grad = grad_fc1_w.detach().clone()
        self.fc1.bias.grad = grad_fc1_b.detach().clone()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        return float(actor_loss.detach().cpu().item())


class Critic(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE + ACTION_SIZE, hidden)
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


class PCCritic(nn.Module):
    """PC-shaped one-hidden-layer Q critic.

    The fast mode trains the same parameters with a normalized MSE loss. The
    exact-local mode uses tick-settled hidden activity plus an explicit output
    error/back-vector update, matching the CartPole exact-local critic story.
    """

    def __init__(self, hidden: int, gamma_pc: float, query_ticks: int, q_scale: float):
        super().__init__()
        self.hidden = hidden
        self.gamma_pc = gamma_pc
        self.query_ticks = query_ticks
        self.q_scale = q_scale
        self.fc1 = nn.Linear(STATE_SIZE + ACTION_SIZE, hidden)
        self.out = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)

    def _input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([states, actions], dim=1)

    def forward_norm(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(self._input(states, actions)))
        return self.out(x)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward_norm(states, actions) * self.q_scale

    def tick_query_norm(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._input(states, actions)
        batch = inputs.shape[0]
        x1 = torch.full((batch, self.hidden), 0.001, dtype=inputs.dtype, device=inputs.device)
        x0 = torch.full((batch, 1), 0.001, dtype=inputs.dtype, device=inputs.device)
        back0 = torch.zeros((batch, self.hidden), dtype=inputs.dtype, device=inputs.device)
        for _ in range(self.query_ticks):
            mu1 = F.linear(inputs, self.fc1.weight, self.fc1.bias)
            eps1 = x1 - mu1
            x1 = x1 + self.gamma_pc * (back0 - eps1)

            hidden_phi = F.relu(x1)
            mu0 = F.linear(hidden_phi, self.out.weight, self.out.bias)
            eps0 = x0 - mu0
            back0 = eps0 @ self.out.weight
            x0 = x0 - self.gamma_pc * eps0
        return x0, x1

    def tick_query(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_norm, _ = self.tick_query_norm(states, actions)
        return q_norm * self.q_scale

    def exactlocal_update(
        self,
        optimizer: torch.optim.Optimizer,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            inputs = self._input(states, actions)
            pred_norm, hidden_state = self.tick_query_norm(states, actions)
            target_norm = targets / self.q_scale
            out_delta = pred_norm - target_norm
            hidden_phi = F.relu(hidden_state)
            hidden_delta = (out_delta @ self.out.weight) * (hidden_state > 0.0).float()
            batch = states.shape[0]

            grad_out_w = (out_delta.T @ hidden_phi) / batch
            grad_out_b = out_delta.mean(dim=0)
            grad_fc1_w = (hidden_delta.T @ inputs) / batch
            grad_fc1_b = hidden_delta.mean(dim=0)
            loss = F.mse_loss(pred_norm * self.q_scale, targets)

        optimizer.zero_grad()
        self.out.weight.grad = grad_out_w.detach().clone()
        self.out.bias.grad = grad_out_b.detach().clone()
        self.fc1.weight.grad = grad_fc1_w.detach().clone()
        self.fc1.bias.grad = grad_fc1_b.detach().clone()
        optimizer.step()
        return float(loss.detach().cpu().item())


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


def evaluate(actor: nn.Module, env_name: str, seed: int, device: torch.device, episodes: int) -> float:
    env = gym.make(env_name)
    rewards = []
    for idx in range(episodes):
        obs, _ = env.reset(seed=seed + 10000 + idx)
        total = 0.0
        done = False
        while not done:
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_t).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def run(args: argparse.Namespace) -> tuple[list[float], list[float], list[float]]:
    set_seed(args.seed)
    device = select_device(args.device)
    if args.print_device:
        print(f"using device: {device}")

    env = gym.make(args.env)
    env.reset(seed=args.seed)
    try:
        env.action_space.seed(args.seed)
    except AttributeError:
        pass

    actor_cls = PCActor if args.actor_backend == "pc" else Actor
    actor = actor_cls(args.hidden).to(device)
    if args.critic_backend == "bp":
        critic = Critic(args.hidden).to(device)
        critic_target = Critic(args.hidden).to(device)
    else:
        critic = PCCritic(args.hidden, args.gamma_pc, args.pc_query, args.pc_critic_q_scale).to(device)
        critic_target = PCCritic(args.hidden, args.gamma_pc, args.pc_query, args.pc_critic_q_scale).to(device)
    actor_target = actor_cls(args.hidden).to(device)
    hard_update(actor_target, actor)
    hard_update(critic_target, critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
    replay = ReplayBuffer(args.replay_size)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rewards: list[float] = []
    eval_rewards: list[float] = []
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    total_steps = 0

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "avg10", "eval_reward",
            "critic_loss", "actor_loss", "steps", "replay_size",
        ])

        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            episode_reward = 0.0
            ep_critic_losses = []
            ep_actor_losses = []
            done = False

            while not done:
                if total_steps < args.start_steps:
                    action = env.action_space.sample()
                else:
                    state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = actor(state_t).squeeze(0).cpu().numpy()
                    action = action + np.random.normal(0.0, args.exploration_noise, size=ACTION_SIZE)
                    action = np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)

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
                        states, actions, batch_rewards, next_states, dones = replay.sample(args.batch_size, device)
                        with torch.no_grad():
                            next_actions = actor_target(next_states)
                            target_q = critic_target(next_states, next_actions)
                            y = batch_rewards + args.discount * (1.0 - dones) * target_q

                        if args.critic_backend == "pc" and args.pc_critic_mode == "exactlocal":
                            critic_loss_value = critic.exactlocal_update(critic_opt, states, actions, y)
                        else:
                            q = critic(states, actions)
                            if args.critic_backend == "pc":
                                critic_loss = F.mse_loss(q / args.pc_critic_q_scale, y / args.pc_critic_q_scale)
                            else:
                                critic_loss = F.mse_loss(q, y)
                            critic_opt.zero_grad()
                            critic_loss.backward()
                            if args.grad_clip > 0:
                                nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                            critic_opt.step()
                            critic_loss_value = float(critic_loss.detach().cpu().item())
                        ep_critic_losses.append(critic_loss_value)

                        if total_steps % args.policy_delay == 0:
                            if args.actor_backend == "pc" and args.pc_actor_mode == "exactlocal":
                                actor_loss_value = actor.exactlocal_update(critic, actor_opt, states, args.grad_clip)
                            else:
                                actor_loss = -critic(states, actor(states)).mean()
                                actor_opt.zero_grad()
                                actor_loss.backward()
                                if args.grad_clip > 0:
                                    nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                                actor_opt.step()
                                actor_loss_value = float(actor_loss.detach().cpu().item())
                            soft_update(actor_target, actor, args.tau)
                            soft_update(critic_target, critic, args.tau)
                            ep_actor_losses.append(actor_loss_value)

            rewards.append(episode_reward)
            avg10 = float(np.mean(rewards[-10:]))
            eval_reward = ""
            if args.eval_every > 0 and episode % args.eval_every == 0:
                eval_reward = evaluate(actor, args.env, args.seed + episode, device, args.eval_episodes)
                eval_rewards.append(float(eval_reward))

            critic_loss = float(np.mean(ep_critic_losses)) if ep_critic_losses else 0.0
            actor_loss = float(np.mean(ep_actor_losses)) if ep_actor_losses else 0.0
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            if episode % args.infotime == 0:
                print({
                    "episode": episode,
                    "reward": episode_reward,
                    "avg10": avg10,
                    "eval_reward": eval_reward,
                    "steps": total_steps,
                }, flush=True)

            writer.writerow([
                episode, episode_reward, avg10, eval_reward,
                critic_loss, actor_loss, total_steps, len(replay),
            ])

    env.close()
    return rewards, critic_losses, actor_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--actor-backend", choices=["bp", "pc"], default="bp")
    parser.add_argument("--pc-actor-mode", choices=["fast", "exactlocal"], default="fast")
    parser.add_argument("--critic-backend", choices=["bp", "pc"], default="bp")
    parser.add_argument("--pc-critic-mode", choices=["fast", "exactlocal"], default="fast")
    parser.add_argument("--pc-critic-q-scale", type=float, default=10.0)
    parser.add_argument("--pc-query", type=int, default=100)
    parser.add_argument("--gamma-pc", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--update-after", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--infotime", type=int, default=10)
    parser.add_argument("--out-csv", default="python_runs/pendulum_ddpg.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--print-device", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    rewards, critic_losses, actor_losses = run(parsed)
    print({
        "episodes": parsed.episodes,
        "final_avg10": float(np.mean(rewards[-10:])) if rewards else 0.0,
        "best": float(np.max(rewards)) if rewards else 0.0,
        "csv": parsed.out_csv,
    })
