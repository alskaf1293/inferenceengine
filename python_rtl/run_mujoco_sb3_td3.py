#!/usr/bin/env python3
"""Stable-Baselines3 TD3 MuJoCo baseline.

This is the conventional RL reference path. Use it to establish that the
environment, reward scale, and standard TD3 machinery solve MuJoCo before
swapping in PC actor/critic components in our own runner.
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed


def evaluate_policy(model: TD3, env_name: str, seed: int, episodes: int) -> float:
    env = gym.make(env_name)
    rewards = []
    for idx in range(episodes):
        obs, _ = env.reset(seed=seed + idx)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


class EvalCsvCallback(BaseCallback):
    def __init__(self, env_name: str, seed: int, eval_freq: int, eval_episodes: int, csv_path: str):
        super().__init__()
        self.env_name = env_name
        self.seed = seed
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.csv_path = csv_path
        self.best_eval = -float("inf")
        self.best_step = 0

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self.file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timesteps", "eval_reward", "best_eval", "best_step"])

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            eval_reward = evaluate_policy(
                self.model,
                self.env_name,
                self.seed + 100000 + self.num_timesteps,
                self.eval_episodes,
            )
            if eval_reward > self.best_eval:
                self.best_eval = eval_reward
                self.best_step = self.num_timesteps
            print({
                "timesteps": self.num_timesteps,
                "eval_reward": eval_reward,
                "best_eval": self.best_eval,
                "best_step": self.best_step,
            }, flush=True)
            self.writer.writerow([self.num_timesteps, eval_reward, self.best_eval, self.best_step])
            self.file.flush()
        return True

    def _on_training_end(self) -> None:
        self.file.close()


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    env = gym.make(args.env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    action_dim = int(np.prod(env.action_space.shape))
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=args.action_noise * np.ones(action_dim),
    )
    policy_kwargs = {"net_arch": [args.hidden, args.hidden]}
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        action_noise=action_noise,
        policy_delay=args.policy_delay,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=0,
    )
    callback = EvalCsvCallback(args.env, args.seed, args.eval_freq, args.eval_episodes, args.out_csv)
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
    final_eval = evaluate_policy(model, args.env, args.seed + 500000, args.final_eval_episodes)
    print({
        "env": args.env,
        "timesteps": args.total_timesteps,
        "final_eval": final_eval,
        "best_eval": callback.best_eval,
        "best_step": callback.best_step,
        "csv": args.out_csv,
    }, flush=True)
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="InvertedPendulum-v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--target-policy-noise", type=float, default=0.2)
    parser.add_argument("--target-noise-clip", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=10)
    parser.add_argument("--out-csv", default="python_runs/mujoco_sb3_td3.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
