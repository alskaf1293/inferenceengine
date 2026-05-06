"""
run_cartpole_millidge_pc.py — Literal Millidge-style CartPole baseline on PC nets.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from millidge_pc_agent import MillidgePCAgent

try:
    import gymnasium as gym
except ImportError:
    import gym


N_EPISODES = 2000
MAX_STEPS = 500
PRINT_EVERY = 50

SEED = 42

AGENT_CONFIG = dict(
    dim_s=4,
    num_actions=2,
    hidden=100,
    discount=0.99,
    gamma_pc=0.1,
    alpha_value=0.001,
    alpha_policy=0.001,
    obs_scale=None,
    N_infer=50,
    N_learn=10,
    N_action=100,
    N_replay_query=100,
    adaptive_inference=True,
    settle_tol=0.001,
    max_infer_ticks=200,
    max_action_ticks=300,
    max_replay_query_ticks=200,
    buffer_size=100000,
    batch_size=200,
    policy_train_every_episodes=5,
    target_update_episodes=50,
    beta=1.0,
    value_target_scale=100.0,
    value_target_clip=5.0,
    policy_target_clip=5.0,
    policy_smoothing=0.02,
    seed=SEED,
)


def run():
    env = gym.make('CartPole-v1')
    env.reset(seed=SEED)
    np.random.seed(SEED)
    agent = MillidgePCAgent(**AGENT_CONFIG)

    os.makedirs('python_runs', exist_ok=True)
    csv_path = 'python_runs/cartpole_millidge_pc.csv'
    results = []
    rolling = []

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'steps', 'avg50', 'value_loss', 'episodes_seen'])
        for ep in range(N_EPISODES):
            obs, _ = env.reset()
            agent.reset()
            prev_reward = 0.0
            steps = 0
            for _ in range(MAX_STEPS):
                action = agent.step(obs, reward=prev_reward, done=False)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                prev_reward = float(reward)
                steps += 1
                if terminated or truncated:
                    agent.step(obs, reward=prev_reward, done=True)
                    break

            results.append(steps)
            rolling.append(steps)
            if len(rolling) > 50:
                rolling.pop(0)
            avg50 = float(np.mean(rolling))
            writer.writerow([ep + 1, steps, f'{avg50:.4f}', f'{agent.last_value_loss:.6f}', agent.episodes_seen])

            if (ep + 1) % PRINT_EVERY == 0:
                print(
                    f'Ep {ep + 1:5d}  steps={steps:4d}  avg50={avg50:6.1f}'
                    f'  value_loss={agent.last_value_loss:.4f}'
                )

    env.close()
    print(f'\nSaved: {csv_path}')
    print(f'Final avg50 steps: {np.mean(results[-50:]):.1f}')
    print(f'Best episode: {max(results)}')


if __name__ == '__main__':
    run()

