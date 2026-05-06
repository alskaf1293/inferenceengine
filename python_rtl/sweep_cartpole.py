"""
sweep_cartpole.py — Focused multi-seed sweep for the PC active inference agent.

This script is intentionally small and opinionated: it sweeps only the knobs
that still matter after the Millidge-style architectural refactor, writes a CSV
summary, and prints the best configs by average final performance.
"""
import csv
import itertools
import os
import sys
from statistics import mean

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from active_inference_agent import ActiveInferenceAgent

try:
    import gymnasium as gym
except ImportError:
    import gym


EPISODES = 100
MAX_STEPS = 500
SEEDS = [7, 21, 42]

GRID = {
    'alpha_efe': [0.0005, 0.0010],
    'alpha_policy': [0.0005, 0.0010],
    'policy_train_every_episodes': [5, 10],
    'policy_batch_size': [32, 64],
    'beta': [0.75, 1.0],
}

BASE_CONFIG = {
    'dim_obs': 4,
    'dim_s': 4,
    'dim_a': 1,
    'hidden_trans': 100,
    'hidden_efe': 100,
    'hidden_policy': 100,
    'gamma_pc': 0.1,
    'alpha_trans': 0.005,
    'discount': 0.99,
    'obs_scale': None,
    'action_candidates': [-1.0, 1.0],
    'N_infer': 20,
    'N_learn': 10,
    'N_action': 50,
    'N_replay': 5,
    'N_replay_query': 20,
    'adaptive_inference': True,
    'settle_tol': 0.001,
    'max_infer_ticks': 100,
    'max_action_ticks': 200,
    'max_replay_query_ticks': 100,
    'buffer_size': 100000,
    'batch_size': 200,
    'target_update_freq': 50,
    'obs_cost_scale': np.array([2.4, 3.0, 0.2095, 2.5]),
    'efe_learn_gamma': 0.0,
    'reset_efe_target_state': True,
}

PROBE_STATES = {
    'safe': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    'danger': np.array([2.0, 0.5, 0.18, 1.0], dtype=np.float64),
    'terminal': np.array([2.4, 0.0, 0.2095, 0.0], dtype=np.float64),
}


def run_trial(config: dict, seed: int) -> dict:
    np.random.seed(seed)
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)

    agent = ActiveInferenceAgent(seed=seed, **config)
    results = []

    for _ in range(EPISODES):
        obs, _ = env.reset()
        agent.reset()
        prev_reward = 0.0
        steps = 0

        for _ in range(MAX_STEPS):
            a_cont = agent.step(obs, reward=prev_reward, done=False)
            action = int(a_cont[0] >= 0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            prev_reward = float(reward)
            steps += 1
            if terminated or truncated:
                agent.step(obs, reward=prev_reward, done=True)
                break

        results.append(steps)

    env.close()

    safe = agent.efe_values_for_state(PROBE_STATES['safe'])
    danger = agent.efe_values_for_state(PROBE_STATES['danger'])
    terminal = agent.efe_values_for_state(PROBE_STATES['terminal'])
    infer_diag = agent.lifetime_inference_diagnostics

    return {
        'seed': seed,
        'avg_last20': float(np.mean(results[-20:])),
        'avg_last50': float(np.mean(results[-50:])),
        'best_episode': int(max(results)),
        'policy_updates': agent.policy_update_count,
        'efe_safe_gap': float(np.max(safe) - np.min(safe)),
        'efe_danger_mean': float(np.mean(danger)),
        'efe_terminal_mean': float(np.mean(terminal)),
        'efe_query_avg_ticks': float(infer_diag['efe_query']['avg_ticks']),
        'efe_target_hit_rate': float(infer_diag['efe_target_query']['max_hit_rate']),
        'policy_query_avg_ticks': float(infer_diag['policy_query']['avg_ticks']),
        'policy_query_hit_rate': float(infer_diag['policy_query']['max_hit_rate']),
    }


def config_rows():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, values))


def main():
    os.makedirs('python_runs', exist_ok=True)
    out_csv = 'python_runs/cartpole_sweep_summary.csv'

    rows = []
    for idx, override in enumerate(config_rows(), start=1):
        config = dict(BASE_CONFIG)
        config.update(override)
        per_seed = [run_trial(config, seed) for seed in SEEDS]
        row = dict(override)
        row['config_id'] = idx
        row['mean_avg_last20'] = mean(item['avg_last20'] for item in per_seed)
        row['mean_avg_last50'] = mean(item['avg_last50'] for item in per_seed)
        row['mean_best_episode'] = mean(item['best_episode'] for item in per_seed)
        row['mean_policy_updates'] = mean(item['policy_updates'] for item in per_seed)
        row['mean_efe_safe_gap'] = mean(item['efe_safe_gap'] for item in per_seed)
        row['mean_efe_danger_mean'] = mean(item['efe_danger_mean'] for item in per_seed)
        row['mean_efe_terminal_mean'] = mean(item['efe_terminal_mean'] for item in per_seed)
        row['mean_efe_query_avg_ticks'] = mean(item['efe_query_avg_ticks'] for item in per_seed)
        row['mean_efe_target_hit_rate'] = mean(item['efe_target_hit_rate'] for item in per_seed)
        row['mean_policy_query_avg_ticks'] = mean(item['policy_query_avg_ticks'] for item in per_seed)
        row['mean_policy_query_hit_rate'] = mean(item['policy_query_hit_rate'] for item in per_seed)
        rows.append(row)
        print(
            f"[{idx:02d}] "
            f"alpha_efe={row['alpha_efe']:.4g} "
            f"alpha_policy={row['alpha_policy']:.4g} "
            f"train_every={row['policy_train_every_episodes']} "
            f"policy_batch={row['policy_batch_size']} "
            f"beta={row['beta']:.2f} "
            f"avg20={row['mean_avg_last20']:.2f} "
            f"best={row['mean_best_episode']:.1f}"
        )

    rows.sort(key=lambda r: r['mean_avg_last20'], reverse=True)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}")
    print("Top configs:")
    for row in rows[:5]:
        print(
            f"  id={row['config_id']:02d} "
            f"avg20={row['mean_avg_last20']:.2f} "
            f"avg50={row['mean_avg_last50']:.2f} "
            f"best={row['mean_best_episode']:.1f} "
            f"alpha_efe={row['alpha_efe']:.4g} "
            f"alpha_policy={row['alpha_policy']:.4g} "
            f"train_every={row['policy_train_every_episodes']} "
            f"policy_batch={row['policy_batch_size']} "
            f"beta={row['beta']:.2f}"
        )


if __name__ == '__main__':
    main()
