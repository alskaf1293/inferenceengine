"""
run_cartpole_millidge_bp.py — direct Python reproduction of Millidge's
DeepActiveInference/active_inference.jl CartPole baseline.

This is intentionally backprop-based. The goal is to mirror the Julia control
loop as closely as possible so we have a trustworthy Python reference before
swapping the substrate back to predictive coding.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
except ImportError:
    import gym


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


class MLP(nn.Module):
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


def sample_action(probs: torch.Tensor) -> int:
    return int(torch.multinomial(probs, num_samples=1).item())


def history_policy_loss(history: History, policynet: MLP, valuenet: MLP,
                        device: torch.device) -> torch.Tensor:
    states = torch.as_tensor(np.asarray(history.states, dtype=np.float32), device=device)
    p = F.softmax(policynet(states), dim=1)
    with torch.no_grad():
        v = valuenet(states)
    return -(p * F.log_softmax(v, dim=1)).sum(dim=1).mean()


def mean_history_policy_loss(histories: list[History], policynet: MLP, valuenet: MLP,
                             device: torch.device) -> torch.Tensor:
    losses = [history_policy_loss(hist, policynet, valuenet, device) for hist in histories if hist.states]
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def replay_expectation(memory: list[tuple[np.ndarray, int, float, np.ndarray, bool]],
                       valuenet: MLP,
                       target_value_net: MLP,
                       policynet: MLP,
                       opt_v: torch.optim.Optimizer,
                       discount: float,
                       device: torch.device) -> float:
    if not memory:
        return 0.0

    batch_size = min(BATCH_SIZE, len(memory))
    minibatch = random.sample(memory, batch_size)

    states = np.stack([item[0] for item in minibatch]).astype(np.float32)
    next_states = np.stack([item[3] for item in minibatch]).astype(np.float32)
    actions = np.asarray([item[1] for item in minibatch], dtype=np.int64)
    rewards = np.asarray([item[2] for item in minibatch], dtype=np.float32)
    dones = np.asarray([item[4] for item in minibatch], dtype=np.bool_)

    x = torch.as_tensor(states, device=device)
    x_next = torch.as_tensor(next_states, device=device)
    rewards_t = torch.as_tensor(rewards, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    dones_t = torch.as_tensor(dones, device=device)

    with torch.no_grad():
        policy_probs = F.softmax(policynet(x_next), dim=1)
        next_values = target_value_net(x_next)
        targets = rewards_t + (~dones_t).float() * discount * (policy_probs * next_values).sum(dim=1)
        y = valuenet(x).detach().clone()
        y[torch.arange(batch_size, device=device), actions_t] = targets

    qhats = valuenet(x)
    loss = F.mse_loss(qhats, y)
    opt_v.zero_grad()
    loss.backward()
    opt_v.step()
    return float(loss.detach().cpu().item())


def main(
    episodes: int = 15000,
    hidden: int = 100,
    discount: float = 0.99,
    lr_policy: float = 0.001,
    lr_value: float = 0.001,
    infotime: int = 50,
    seed: int = 42,
    out_csv: str = 'python_runs/cartpole_millidge_bp.csv',
) -> tuple[list[int], list[float], list[float]]:
    set_seed(seed)
    device = torch.device('cpu')
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)

    valuenet = MLP(STATE_SIZE, hidden, ACTION_SIZE).to(device)
    policynet = MLP(STATE_SIZE, hidden, ACTION_SIZE).to(device)
    target_value_net = MLP(STATE_SIZE, hidden, ACTION_SIZE).to(device)
    target_value_net.load_state_dict(valuenet.state_dict())

    opt_p = torch.optim.Adam(list(valuenet.parameters()) + list(policynet.parameters()), lr=lr_policy)
    opt_v = torch.optim.Adam(valuenet.parameters(), lr=lr_value)

    avgreward = 0.0
    histories: list[History] = []
    ep_rewards: list[int] = []
    plosses: list[float] = []
    vlosses: list[float] = []
    memory: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'avg_reward_ema', 'policy_loss', 'value_loss'])

        for episode in range(1, episodes + 1):
            obs, _ = env.reset()
            state = np.asarray(obs, dtype=np.float32)
            episode_rewards = 0
            history = History(STATE_SIZE, ACTION_SIZE, discount)

            for _ in range(10000):
                state_t = torch.as_tensor(state, device=device).unsqueeze(0)
                probs = F.softmax(policynet(state_t), dim=1).squeeze(0)
                action = sample_action(probs)

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
                episode_rewards += reward
                if done:
                    break

            histories.append(history)
            avgreward = 0.1 * episode_rewards + 0.9 * avgreward

            if episode % infotime == 0:
                print(f'(episode:{episode}, avgreward:{avgreward})')

            if episode % 5 == 0:
                ploss = mean_history_policy_loss(histories, policynet, valuenet, device)
                opt_p.zero_grad()
                ploss.backward()
                opt_p.step()
                histories = []
            else:
                ploss = history_policy_loss(history, policynet, valuenet, device)

            if episode % 50 == 0:
                target_value_net.load_state_dict(valuenet.state_dict())

            vloss = replay_expectation(memory, valuenet, target_value_net, policynet, opt_v, discount, device)

            ep_rewards.append(int(episode_rewards))
            plosses.append(float(ploss.detach().cpu().item()))
            vlosses.append(vloss)
            writer.writerow([episode, episode_rewards, avgreward, plosses[-1], vloss])

    env.close()
    return ep_rewards, plosses, vlosses


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=15000)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--discount', type=float, default=0.99)
    ap.add_argument('--lr-policy', type=float, default=0.001)
    ap.add_argument('--lr-value', type=float, default=0.001)
    ap.add_argument('--infotime', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-csv', type=str, default='python_runs/cartpole_millidge_bp.csv')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rewards, plosses, vlosses = main(
        episodes=args.episodes,
        hidden=args.hidden,
        discount=args.discount,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        infotime=args.infotime,
        seed=args.seed,
        out_csv=args.out_csv,
    )
    print({
        'final_avg50': float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards)),
        'best': int(max(rewards)) if rewards else 0,
        'episodes': len(rewards),
        'csv': args.out_csv,
    })
