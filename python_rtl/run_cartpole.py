"""
run_cartpole.py — CartPole-v1 validation for the deep active inference agent.

Implements Millidge (2019) "Deep Active Inference as Variational Policy Gradients".
Action selection: a PC policy network trained from EFE-implied action preferences.

Output: python_runs/cartpole_ai.csv   columns: episode,steps,G_estimate,trans_err
"""
import numpy as np
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from active_inference_agent import ActiveInferenceAgent

try:
    import gymnasium as gym
except ImportError:
    import gym

# ── hyperparameters ────────────────────────────────────────────────────────────
N_EPISODES     = 2000
MAX_STEPS      = 500        # CartPole-v1 episode cap
PRINT_EVERY    = 50
PROBE_EVERY    = 10

# Agent parameters — Millidge (2019) algorithm with PC networks
# Reduced ticks vs paper's ADAM (PC needs iterate; ADAM single pass)
DIM_OBS        = 4
DIM_S          = 4
DIM_A          = 1
HIDDEN_TRANS   = 100
HIDDEN_EFE     = 100
HIDDEN_POLICY  = 100
GAMMA_PC       = 0.1        # PC inference rate
ALPHA_TRANS    = 0.005
ALPHA_EFE      = 0.001
ALPHA_POLICY   = 0.001
DISCOUNT       = 0.99
N_INFER        = 20         # settle ticks (α=0) for online learn
N_LEARN        = 10         # weight ticks (α>0) for online learn
N_ACTION       = 50         # ticks per EFE query; G*≈-66.7 at convergence
N_REPLAY       = 5          # weight ticks per replay sample
N_REPLAY_QUERY = 20         # target query ticks at replay time (< N_ACTION for speed)
ADAPTIVE_INFERENCE = True
SETTLE_TOL     = 0.001
MAX_INFER_TICKS = 100
MAX_ACTION_TICKS = 200
MAX_REPLAY_QUERY_TICKS = 100
BETA           = 1.0        # softer policy while EFE estimates are still noisy
BUFFER_SIZE    = 100000     # closer to Millidge replay scale
BATCH_SIZE     = 200        # Millidge-style replay batch size
TARGET_UPDATE  = 50         # EFE target network sync period (episodes)
RESET_EFE_TARGET_STATE = True
POLICY_TRAIN_EVERY_EPISODES = 5
POLICY_BATCH_SIZE = 64      # smaller PC policy refresh approximates one batch update better
POLICY_FALLBACK_ON_UNSETTLED = True

# No obs preprocessing for network inputs (Millidge 2019: "No preprocessing")
OBS_SCALE      = None

# Observation cost scale: CartPole termination boundaries used as Gaussian prior σ.
# Cost = 0.5 * sum((obs / OBS_COST_SCALE)²)  →  0 at goal, ~0.5 at each boundary.
# Replaces flat r=1 with a state-dependent cost that immediately discriminates
# balanced from near-failure states without needing many lucky long episodes.
OBS_COST_SCALE = np.array([2.4, 3.0, 0.2095, 2.5])

ACTION_CANDIDATES = [-1.0, 1.0]

SEED = 42

PROBE_STATES = {
    'safe': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    'danger': np.array([2.0, 0.5, 0.18, 1.0], dtype=np.float64),
    'terminal': np.array([2.4, 0.0, 0.2095, 0.0], dtype=np.float64),
}


def run():
    env = gym.make('CartPole-v1')
    env.reset(seed=SEED)
    np.random.seed(SEED)

    agent = ActiveInferenceAgent(
        dim_obs          = DIM_OBS,
        dim_s            = DIM_S,
        dim_a            = DIM_A,
        hidden_trans     = HIDDEN_TRANS,
        hidden_efe       = HIDDEN_EFE,
        hidden_policy    = HIDDEN_POLICY,
        gamma_pc         = GAMMA_PC,
        alpha_trans      = ALPHA_TRANS,
        alpha_efe        = ALPHA_EFE,
        alpha_policy     = ALPHA_POLICY,
        discount         = DISCOUNT,
        obs_scale        = OBS_SCALE,
        action_candidates= ACTION_CANDIDATES,
        N_infer          = N_INFER,
        N_learn          = N_LEARN,
        N_action         = N_ACTION,
        N_replay         = N_REPLAY,
        N_replay_query   = N_REPLAY_QUERY,
        adaptive_inference = ADAPTIVE_INFERENCE,
        settle_tol       = SETTLE_TOL,
        max_infer_ticks  = MAX_INFER_TICKS,
        max_action_ticks = MAX_ACTION_TICKS,
        max_replay_query_ticks = MAX_REPLAY_QUERY_TICKS,
        beta             = BETA,
        buffer_size      = BUFFER_SIZE,
        batch_size       = BATCH_SIZE,
        target_update_freq = TARGET_UPDATE,
        obs_cost_scale   = OBS_COST_SCALE,
        efe_learn_gamma  = 0.0,
        reset_efe_target_state = RESET_EFE_TARGET_STATE,
        policy_fallback_on_unsettled = POLICY_FALLBACK_ON_UNSETTLED,
        policy_train_every_episodes = POLICY_TRAIN_EVERY_EPISODES,
        policy_batch_size = POLICY_BATCH_SIZE,
        seed             = SEED,
    )

    os.makedirs('python_runs', exist_ok=True)
    csv_path = 'python_runs/cartpole_ai.csv'

    results = []
    rolling = []

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'steps', 'G_estimate', 'trans_err', 'policy_updates',
            'policy_action_fallbacks', 'policy_bootstrap_fallbacks',
            'efe_query_avg_ticks', 'efe_query_max_hit_rate',
            'efe_target_query_avg_ticks', 'efe_target_query_max_hit_rate',
            'policy_query_avg_ticks', 'policy_query_max_hit_rate',
            'efe_online_settle_avg_ticks', 'efe_online_settle_final_delta',
            'efe_replay_settle_avg_ticks', 'efe_replay_settle_final_delta',
            'policy_settle_avg_ticks', 'policy_settle_final_delta',
            'efe_safe_0', 'efe_safe_1', 'efe_danger_0', 'efe_danger_1',
            'efe_terminal_0', 'efe_terminal_1',
            'policy_safe_0', 'policy_safe_1',
            'bootstrap_safe_0', 'bootstrap_safe_1',
        ])

        for ep in range(N_EPISODES):
            obs, _ = env.reset()
            agent.reset()

            prev_reward = 0.0
            steps = 0

            for t in range(MAX_STEPS):
                a_cont = agent.step(obs, reward=prev_reward, done=False)

                # Map continuous action to discrete CartPole action
                discrete_action = int(a_cont[0] >= 0.0)

                obs, reward, terminated, truncated, _ = env.step(discrete_action)
                prev_reward = float(reward)
                steps += 1

                if terminated or truncated:
                    # Terminal update: pass done=True so G_prev is zeroed
                    agent.step(obs, reward=prev_reward, done=True)
                    break

            G_est = agent.G_estimate
            t_err = agent.transition_prediction_error
            policy_updates = agent.policy_update_count
            policy_action_fallbacks = agent.policy_action_fallback_count
            policy_bootstrap_fallbacks = agent.policy_bootstrap_fallback_count
            infer_diag = agent.inference_diagnostics
            efe_query = infer_diag['efe_query']
            efe_target_query = infer_diag['efe_target_query']
            policy_query = infer_diag['policy_query']
            efe_online_settle = infer_diag['efe_online_settle']
            efe_replay_settle = infer_diag['efe_replay_settle']
            policy_settle = infer_diag['policy_settle']
            results.append(steps)
            rolling.append(steps)
            if len(rolling) > 50:
                rolling.pop(0)

            probe_row = [''] * 10
            if (ep + 1) % PROBE_EVERY == 0:
                efe_safe = agent.efe_values_for_state(PROBE_STATES['safe'])
                efe_danger = agent.efe_values_for_state(PROBE_STATES['danger'])
                efe_terminal = agent.efe_values_for_state(PROBE_STATES['terminal'])
                policy_safe = agent.policy_probs_for_state(PROBE_STATES['safe'])
                boot_safe = agent.bootstrap_policy_probs_for_state(PROBE_STATES['safe'])
                probe_row = [
                    f'{efe_safe[0]:.4f}', f'{efe_safe[1]:.4f}',
                    f'{efe_danger[0]:.4f}', f'{efe_danger[1]:.4f}',
                    f'{efe_terminal[0]:.4f}', f'{efe_terminal[1]:.4f}',
                    f'{policy_safe[0]:.4f}', f'{policy_safe[1]:.4f}',
                    f'{boot_safe[0]:.4f}', f'{boot_safe[1]:.4f}',
                ]

            writer.writerow([
                ep, steps, f'{G_est:.4f}', f'{t_err:.4f}', policy_updates,
                policy_action_fallbacks, policy_bootstrap_fallbacks,
                f"{efe_query['avg_ticks']:.2f}", f"{efe_query['max_hit_rate']:.4f}",
                f"{efe_target_query['avg_ticks']:.2f}", f"{efe_target_query['max_hit_rate']:.4f}",
                f"{policy_query['avg_ticks']:.2f}", f"{policy_query['max_hit_rate']:.4f}",
                f"{efe_online_settle['avg_ticks']:.2f}", f"{efe_online_settle['avg_final_delta']:.4f}",
                f"{efe_replay_settle['avg_ticks']:.2f}", f"{efe_replay_settle['avg_final_delta']:.4f}",
                f"{policy_settle['avg_ticks']:.2f}", f"{policy_settle['avg_final_delta']:.4f}",
            ] + probe_row)

            if (ep + 1) % PRINT_EVERY == 0:
                avg = np.mean(rolling)
                print(f'Ep {ep+1:5d}  steps={steps:4d}  avg50={avg:6.1f}'
                      f'  trans_err={t_err:.4f}  G={G_est:+.3f}'
                      f'  policy_updates={policy_updates}'
                      f'  policy_fallbacks={policy_action_fallbacks}'
                      f"  efe_q={efe_query['avg_ticks']:.1f}/{MAX_ACTION_TICKS}"
                      f"  pol_q={policy_query['avg_ticks']:.1f}/{MAX_ACTION_TICKS}"
                      f"  tgt_q_hit={efe_target_query['max_hit_rate']:.2f}")

    env.close()
    print(f'\nSaved: {csv_path}')
    print(f'Final avg50 steps: {np.mean(results[-50:]):.1f}  (target: 500)')
    print(f'Best episode: {max(results)} steps')


if __name__ == '__main__':
    run()
