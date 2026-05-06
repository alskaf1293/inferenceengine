"""
diagnose_cartpole_equivalence.py — isolate substrate vs outer-loop failure.

This script answers two questions:

1. Convergence gap:
   How much do EFE values and policy logits still drift if we allow more than
   the default number of inference ticks?

2. Frozen-target fit:
   If we freeze a CartPole state -> EFE-vector dataset, can a PC network with
   the same topology fit it comparably to a backprop MLP?
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from active_inference_agent import ActiveInferenceAgent

try:
    import gymnasium as gym
except ImportError:
    import gym


@dataclass
class BackpropMLP:
    n_in: int
    n_hidden: int
    n_out: int
    lr: float
    seed: int

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.standard_normal((self.n_hidden, self.n_in)) * 0.1
        self.b1 = np.zeros(self.n_hidden, dtype=np.float64)
        self.W2 = rng.standard_normal((self.n_out, self.n_hidden)) * 0.1
        self.b2 = np.zeros(self.n_out, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z1 = self.W1 @ x + self.b1
        self.h = np.tanh(self.z1)
        self.y = self.W2 @ self.h + self.b2
        return self.y

    def update(self, y_target: np.ndarray) -> None:
        dy = 2.0 * (self.y - y_target) / len(y_target)
        dW2 = np.outer(dy, self.h)
        db2 = dy
        dh = self.W2.T @ dy
        dz1 = dh * (1.0 - np.tanh(self.z1) ** 2)
        dW1 = np.outer(dz1, self.x)
        db1 = dz1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2


def make_agent(seed: int) -> ActiveInferenceAgent:
    return ActiveInferenceAgent(
        dim_obs=4,
        dim_s=4,
        dim_a=1,
        hidden_trans=100,
        hidden_efe=100,
        hidden_policy=100,
        gamma_pc=0.1,
        alpha_trans=0.005,
        alpha_efe=0.0005,
        alpha_policy=0.001,
        discount=0.99,
        obs_scale=None,
        action_candidates=[-1.0, 1.0],
        N_infer=20,
        N_learn=10,
        N_action=50,
        N_replay=5,
        N_replay_query=20,
        adaptive_inference=True,
        settle_tol=0.001,
        max_infer_ticks=100,
        max_action_ticks=200,
        max_replay_query_ticks=100,
        beta=1.0,
        buffer_size=100000,
        batch_size=200,
        target_update_freq=50,
        obs_cost_scale=np.array([2.4, 3.0, 0.2095, 2.5]),
        efe_learn_gamma=0.0,
        reset_efe_target_state=True,
        policy_train_every_episodes=5,
        policy_batch_size=64,
        seed=seed,
    )


def train_agent(agent: ActiveInferenceAgent, episodes: int, max_steps: int, seed: int) -> None:
    np.random.seed(seed)
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    for _ in range(episodes):
        obs, _ = env.reset()
        agent.reset()
        prev_reward = 0.0
        for _ in range(max_steps):
            a_cont = agent.step(obs, reward=prev_reward, done=False)
            action = int(a_cont[0] >= 0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            prev_reward = float(reward)
            if terminated or truncated:
                agent.step(obs, reward=prev_reward, done=True)
                break
    env.close()


def collect_states(num_states: int, max_steps: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    states = []
    obs, _ = env.reset()
    while len(states) < num_states:
        states.append(np.asarray(obs[:4], dtype=np.float64))
        action = np.random.randint(0, 2)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.asarray(states, dtype=np.float64)


def mse_pc(net, x_data: np.ndarray, y_data: np.ndarray, gamma: float, settle: int) -> float:
    net.set_rates(alpha=0.0, gamma=gamma)
    acc = 0.0
    for x, y in zip(x_data, y_data):
        net.reset_state()
        for _ in range(settle):
            net.tick(x, y_bottom=None, clamp_top=True, clamp_bottom=False)
        diff = net.x0 - y
        acc += float(np.mean(diff * diff))
    return acc / len(x_data)


def mse_bp(mlp: BackpropMLP, x_data: np.ndarray, y_data: np.ndarray) -> float:
    acc = 0.0
    for x, y in zip(x_data, y_data):
        diff = mlp.predict(x) - y
        acc += float(np.mean(diff * diff))
    return acc / len(x_data)


def run_frozen_target_compare(agent: ActiveInferenceAgent,
                              train_states: np.ndarray,
                              test_states: np.ndarray,
                              epochs: int,
                              seed: int) -> None:
    from pc_network import PCNet3Layer

    y_train = np.asarray([agent.efe_values_for_state(s) for s in train_states], dtype=np.float64)
    y_test = np.asarray([agent.efe_values_for_state(s) for s in test_states], dtype=np.float64)

    pc = PCNet3Layer(
        k_lut=[2, 100, 4],
        act_lut=['linear', 'tanh', 'linear'],
        wclip=20.0,
        gamma=0.1,
        alpha=0.0005,
        xclip_lut=[2.0, 10.0, None],
        eps_clip_lut=[1.0, 1.0, 1.0],
        seed=seed,
        rtl_init=False,
        gen_k_lut=None,
    )
    bp = BackpropMLP(4, 100, 2, lr=0.0005, seed=seed)

    print("\nFrozen-target fit")
    for ep in range(epochs + 1):
        if ep > 0:
            for x, y in zip(train_states, y_train):
                pc.reset_state()
                pc.set_rates(alpha=0.0, gamma=0.1)
                for _ in range(20):
                    pc.tick(x, y, clamp_top=True, clamp_bottom=True)
                pc.set_rates(alpha=0.0005, gamma=0.0)
                for _ in range(10):
                    pc.tick(x, y, clamp_top=True, clamp_bottom=True)

                bp.forward(x)
                bp.update(y)

        pc_mse = mse_pc(pc, test_states, y_test, gamma=0.1, settle=50)
        bp_mse = mse_bp(bp, test_states, y_test)
        print(f"epoch={ep:02d} pc_mse={pc_mse:.6f} bp_mse={bp_mse:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-episodes', type=int, default=20)
    ap.add_argument('--train-states', type=int, default=256)
    ap.add_argument('--test-states', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    agent = make_agent(args.seed)
    print(f"Training reference PC agent for {args.train_episodes} episodes...")
    train_agent(agent, episodes=args.train_episodes, max_steps=500, seed=args.seed)
    print("\nAdaptive inference stats")
    for name, stats in agent.lifetime_inference_diagnostics.items():
        if stats['calls'] <= 0:
            continue
        print(
            f"  {name:18s} calls={int(stats['calls']):5d}"
            f" avg_ticks={stats['avg_ticks']:.2f}"
            f" avg_extra={stats['avg_extra_ticks']:.2f}"
            f" final_delta={stats['avg_final_delta']:.4f}"
            f" max_hit_rate={stats['max_hit_rate']:.3f}"
        )

    probes = {
        'safe': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        'danger': np.array([2.0, 0.5, 0.18, 1.0], dtype=np.float64),
        'terminal': np.array([2.4, 0.0, 0.2095, 0.0], dtype=np.float64),
    }

    print("\nConvergence gap")
    for name, obs in probes.items():
        profile = agent.convergence_profile_for_state(obs)
        print(name)
        for tick, efe, pol in zip(profile['ticks'], profile['efe_values'], profile['policy_logits']):
            efe_fmt = ', '.join(f'{v:.3f}' for v in efe)
            pol_fmt = ', '.join(f'{v:.3f}' for v in pol)
            print(f"  ticks={tick:3d} efe=[{efe_fmt}] policy=[{pol_fmt}]")

    train_states = collect_states(args.train_states, max_steps=500, seed=args.seed + 1)
    test_states = collect_states(args.test_states, max_steps=500, seed=args.seed + 2)
    run_frozen_target_compare(agent, train_states, test_states, epochs=args.epochs, seed=args.seed)


if __name__ == '__main__':
    main()
