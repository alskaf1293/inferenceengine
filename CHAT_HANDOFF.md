# Chat Handoff

## Goal

Reproduce Millidge (2019) CartPole active inference behavior while ultimately replacing backprop-trained networks with predictive-coding networks under the Whittington-style PC/backprop equivalence.

## What Was Added

### Reference and Comparison Runners

- `python_rtl/run_cartpole_millidge_bp.py`
  - Direct Python backprop reference for the Julia `DeepActiveInference/active_inference.jl` baseline.
  - Important fixes:
    - policy/history loss uses detached value outputs, matching Julia `V.data`
    - MLP init uses Xavier/Glorot uniform with zero bias to better match Flux `Dense`
- `python_rtl/run_cartpole_millidge_hybrid.py`
  - Millidge-style outer loop with independently swappable `value_backend` and `policy_backend`
  - supports `bp` or `pc` per network
  - used to replace backprop networks one by one
- `python_rtl/run_cartpole_millidge_pc.py`
  - early literal-PC branch that was superseded by the hybrid runner

### PC / Diagnosis Tools

- `python_rtl/active_inference_agent.py`
  - hybrid active-inference PC agent used before the stricter Millidge reference path
- `python_rtl/diagnose_cartpole_equivalence.py`
  - convergence-gap and frozen-target diagnostic script
- `python_rtl/millidge_pc_agent.py`
  - literal-PC baseline branch used before the hybrid runner
- `python_rtl/sweep_cartpole.py`
  - sweep runner for the older hybrid PC agent

## Most Important Experimental Findings

### 1. Backprop Reference Works

`python_rtl/run_cartpole_millidge_bp.py` now reaches the right qualitative regime.

At 2000 episodes:

- seed `1`: `avg50 = 310.12`, `best = 500`
- seed `7`: `avg50 = 294.46`, `best = 500`
- seed `21`: `avg50 = 215.26`, `best = 500`
- seed `42`: `avg50 = 269.00`, `best = 500`
- seed `84`: `avg50 = 263.02`, `best = 500`

Mean over those five seeds:

- `mean_avg50 = 270.37`

So the Python translation is good enough to use as the working Millidge-style baseline.

### 2. First One-by-One Replacement Is Live

The first actual replacement path is:

- `value_backend = pc`
- `policy_backend = bp`

That path runs end-to-end in `python_rtl/run_cartpole_millidge_hybrid.py`.

Observed early results:

- `bp/bp`, 50 episodes: `final_avg50 = 22.14`, `best = 96`
- `pc/bp`, 50 episodes: `final_avg50 = 21.22`, `best = 73`

So the first swapped branch is stable, but it does not yet retain the long-horizon backprop result.

### 3. Important PC Bug Was Fixed

In the hybrid PC path, the PC learners were zeroing their own learning rate before the update ticks.

This was fixed in:

- `python_rtl/run_cartpole_millidge_hybrid.py`

The bug was real, but fixing it alone was not enough to close the backprop gap.

### 4. Policy Settling Was a Major Problem in the Older Hybrid PC Agent

Earlier instrumentation on `python_rtl/active_inference_agent.py` showed:

- EFE settle/query paths were mostly reasonable
- the amortized policy network often needed far more settling than budgeted

This motivated policy fallback logic in the older agent, but the cleaner current path is to work through the Millidge hybrid runner and swap networks one by one.

## Current Working Theory

The remaining gap is not that the whole project is impossible. It is that the first swapped PC value network is not yet matching the backprop replay gradient inside the Millidge loop.

New diagnostic result from May 7, 2026:

- The PC value network's free forward query can match the equivalent BP MLP when given enough query ticks.
- The original full-clamp local PC value update does **not** match the BP gradient:
  - frozen random-batch critic gradient cosine was about `0.35`
  - increasing inference ticks did not fix it
- A diagnostic `bp_equiv` gradient mode was added to `python_rtl/run_cartpole_millidge_hybrid.py`.
  - It keeps the PC value network representation/target/query path, but uses the exact two-layer BP-equivalent gradient for the value update.
  - Frozen-batch gradient cosine vs PyTorch BP is effectively `1.0`.
  - In CartPole, `pc/bp --pc-gradient-mode bp_equiv` tracks the `bp/bp` baseline exactly through 200 episodes for seed `42`.
- A more PC-native experimental mode, `pc_nudge_gated`, was also added.
  - It uses a small output nudge and gates the hidden update by the downstream ReLU derivative.
  - Frozen-batch gradient cosine improves to about `0.98`.
  - First 50-episode seed-42 probe: `final_avg50 = 22.76`, `best = 102`.
- A CUDA batched bridge mode, `pc_nudge_gated_fast`, was added after the GPU became available.
  - It keeps the PC critic replacement structure but batches the derivative-gated critic update in Torch/CUDA.
  - This is for efficient long sweeps; it is not the RTL-tick-faithful NumPy settling path.
  - 500-episode checks exactly matched `bp/bp` across seeds `1, 7, 21, 42, 84`.
  - 2000-episode `pc/bp` fast sweep over seeds `1, 7, 21, 42, 84`:
    - seed `1`: `avg50 = 406.78`, `best = 500`
    - seed `7`: `avg50 = 279.04`, `best = 500`
    - seed `21`: `avg50 = 230.08`, `best = 500`
    - seed `42`: `avg50 = 343.20`, `best = 500`
    - seed `84`: `avg50 = 246.68`, `best = 500`
    - mean `avg50 = 301.16`
  - Matching hybrid `bp/bp` seed `42` at 2000 episodes: `avg50 = 313.30`, `best = 500`.
- A CUDA fast bridge for the PC policy was also added:
  - use `--policy-backend pc --pc-policy-gradient-mode fast`
  - this keeps the PC policy's two-layer structure but batches the history update in Torch/CUDA.
  - 500-episode `pc/pc` checks exactly matched `bp/bp`/`pc/bp` across seeds `1, 7, 21, 42, 84`.
  - 2000-episode `pc/pc` fast sweep over seeds `1, 7, 21, 42, 84`:
    - seed `1`: `avg50 = 406.78`, `best = 500`
    - seed `7`: `avg50 = 279.04`, `best = 500`
    - seed `21`: `avg50 = 230.08`, `best = 500`
    - seed `42`: `avg50 = 343.20`, `best = 500`
    - seed `84`: `avg50 = 246.68`, `best = 500`
    - mean `avg50 = 301.16`
  - This is the first complete CartPole `pc/pc` active-inference bridge result.
- A reproducible CartPole bridge report script was added:
  - `scripts/exp_cartpole_pc_bridge.py`
  - runs missing sweeps with `--run-missing`
  - writes `python_runs/cartpole_bridge_summary.csv`
  - writes `python_runs/cartpole_bridge_aggregate.csv`
  - writes `figures/cartpole_bridge_learning_curves.png`
  - latest aggregate over seeds `1, 7, 21, 42, 84`:
    - `bp/bp`: mean `avg50 = 285.64`, mean `avg100 = 287.26`, mean `best = 500`
    - `pc/bp fast`: mean `avg50 = 301.16`, mean `avg100 = 301.48`, mean `best = 500`
    - `pc/pc fast`: mean `avg50 = 301.16`, mean `avg100 = 301.48`, mean `best = 500`
- A bridge-vs-RTL-faithful alignment diagnostic was added:
  - `scripts/diagnose_pc_bridge_equivalence.py`
  - writes `python_runs/pc_bridge_equivalence.csv`
  - writes `figures/pc_bridge_gradient_alignment.png`
  - optional short CartPole probes write `python_runs/pc_bridge_short_cartpole.csv`
  - latest frozen-batch result:
    - original full-clamp `pc`: gradient cosine vs BP `0.337`, relative error `1.004`
    - tick-faithful `pc_nudge_gated`: gradient cosine vs BP `0.990`, relative error `0.515`
    - Torch tick-faithful `pc_nudge_gated_torch_tick`: gradient cosine vs BP `0.990`, relative error `0.515`
    - CUDA bridge `pc_nudge_gated_fast`: gradient cosine vs BP `1.000`, relative error `~1e-7`
    - exact `bp_equiv_fast`: gradient cosine vs BP `1.000`, relative error `~1e-7`
    - forward-query MSE vs BP is essentially zero for all modes at 300 query ticks
  - latest 50-episode seed-42 short CartPole probe:
    - `pc`: `avg50 = 21.06`, `best = 70`
    - `pc_nudge_gated`: `avg50 = 22.76`, `best = 102`
    - `pc_nudge_gated_fast`: `avg50 = 20.90`, `best = 59`
  - interpretation: the derivative-gated/nudged update is the right local tick direction, but the tick-faithful magnitude/optimizer scaling still needs alignment with the exact fast bridge.
- A first short `pc/pc` alignment run was completed for seed `42`, 50 episodes:
  - fast bridge: `--pc-gradient-mode pc_nudge_gated_fast --pc-policy-gradient-mode fast`
    - `avg50 = 20.90`, `best = 59`
  - Torch tick-faithful: `--pc-gradient-mode pc_nudge_gated_torch_tick --pc-policy-gradient-mode torch_tick`
    - `avg50 = 24.60`, `best = 67`
  - NumPy tick-faithful: `--pc-gradient-mode pc_nudge_gated --pc-policy-gradient-mode pc --pc-infer 300 --max-infer-ticks 300 --no-adaptive-inference`
    - `avg50 = 23.40`, `best = 102`
  - interpretation: Torch tick is behaviorally in the same short-run regime as NumPy tick and much faster, so it is a viable alignment lane. Need more seeds / longer runs before treating it as equivalent.
- A five-seed short `pc/pc` alignment report was then added and run:
  - script: `scripts/exp_cartpole_pc_pc_alignment.py`
  - summary: `python_runs/pc_pc_alignment_summary_50.csv`
  - aggregate: `python_runs/pc_pc_alignment_aggregate_50.csv`
  - seeds: `1, 7, 21, 42, 84`
  - episodes: `50`
  - aggregate result:
    - CUDA fast bridge: mean `avg50 = 21.14`, std `0.99`, mean `best = 57.4`, max `best = 82`
    - Torch tick-faithful: mean `avg50 = 21.94`, std `3.86`, mean `best = 54.6`, max `best = 67`
    - NumPy tick-faithful: mean `avg50 = 21.62`, std `1.25`, mean `best = 70.6`, max `best = 102`
  - interpretation: over short early-learning probes, the GPU fast bridge is not obviously hiding a semantic mismatch. Torch tick and NumPy tick sit in the same behavioral band. The next required test is longer multi-seed alignment, where optimizer/magnitude differences should become more visible.
- A 200-episode `pc/pc` alignment was started next.
  - full five-seed fast bridge and Torch tick-faithful lanes completed
  - full five-seed NumPy tick-faithful lane was stopped because it is too slow for interactive 200-episode sweeps
  - `scripts/exp_cartpole_pc_pc_alignment.py` now supports `--modes` so CUDA sweeps and targeted NumPy spot checks can be separated
  - summary: `python_runs/pc_pc_alignment_summary_200.csv`
  - aggregate: `python_runs/pc_pc_alignment_aggregate_200.csv`
  - fast bridge over seeds `1, 7, 21, 42, 84`:
    - mean episode reward `19.08`
    - mean final `avg50 = 15.72`
    - mean `best = 68.4`, max `best = 82`
  - Torch tick-faithful over seeds `1, 7, 21, 42, 84`:
    - mean episode reward `21.58`
    - mean final `avg50 = 20.18`
    - mean `best = 86.8`, max `best = 124`
  - interpretation: Torch tick is not collapsing relative to the fast bridge at 200 episodes; if anything, it is stronger in this early/transient window. The remaining missing piece is targeted NumPy tick spot checks at 200 episodes, not full NumPy sweeps.
- A targeted 200-episode NumPy tick spot check was completed for seed `42`.
  - `scripts/exp_cartpole_pc_pc_alignment.py` now supports `--tag` so spot checks do not overwrite broad reports.
  - spot-check summary: `python_runs/pc_pc_alignment_summary_200_seed42_spotcheck.csv`
  - spot-check aggregate: `python_runs/pc_pc_alignment_aggregate_200_seed42_spotcheck.csv`
  - seed `42`, 200 episodes:
    - CUDA fast bridge: mean reward `17.38`, final `avg50 = 13.76`, `best = 59`
    - Torch tick-faithful: mean reward `22.67`, final `avg50 = 20.86`, `best = 94`
    - NumPy tick-faithful: mean reward `23.39`, final `avg50 = 24.66`, `best = 102`
  - interpretation: on the longer seed-42 reference, Torch tick is much closer to NumPy tick than the fast bridge is. This is the strongest evidence so far that the GPU tick-faithful lane is preserving the local PC semantics rather than just acting as another BP bridge.
- A 500-episode Torch tick-faithful `pc/pc` five-seed run was completed.
  - command mode: `--pc-gradient-mode pc_nudge_gated_torch_tick --pc-policy-gradient-mode torch_tick --device cuda`
  - summary: `python_runs/pc_pc_alignment_summary_500_torch_tick_5seed.csv`
  - aggregate: `python_runs/pc_pc_alignment_aggregate_500_torch_tick_5seed.csv`
  - seeds `1, 7, 21, 42, 84`
  - mean reward `19.65`
  - mean final `avg50 = 18.55`
  - mean `best = 87.6`, max `best = 124`
  - interpretation: the RTL-faithful GPU lane is stable and usable for longer runs, but it does **not** yet reproduce the solved Millidge-style CartPole curve. This is now an optimizer/magnitude alignment problem rather than a GPU/runtime problem.
- A policy derivative-gated Torch tick variant was added:
  - policy mode: `--pc-policy-gradient-mode torch_tick_gated`
  - report mode: `pc_pc_torch_tick_gated`
  - seed `42`, 200 episodes:
    - gated Torch tick mean reward `23.23`, final `avg50 = 19.80`, `best = 100`
    - NumPy tick mean reward `23.39`, final `avg50 = 24.66`, `best = 102`
  - five-seed 500-episode gated policy run:
    - summary: `python_runs/pc_pc_alignment_summary_500_torch_tick_gated_5seed.csv`
    - aggregate: `python_runs/pc_pc_alignment_aggregate_500_torch_tick_gated_5seed.csv`
    - mean reward `19.06`
    - mean final `avg50 = 16.24`
    - mean `best = 76.2`, max `best = 100`
  - interpretation: policy derivative gating is semantically plausible and matches the seed-42 200-episode NumPy mean closely, but it does not improve the 500-episode five-seed result. Keep it as an experimental RTL option, not the default.
- Quick 500-episode seed-42 Torch tick tuning probes:
  - baseline default: final `avg50 = 26.82`, `best = 94`
  - replay updates x4: final `avg50 = 19.62`, `best = 88`
  - `lr-value = lr-policy = 0.003`: final `avg50 = 10.80`, `best = 81`
  - `lr-value = lr-policy = 0.0003`: final `avg50 = 22.88`, `best = 101`
  - `pc-nudge-beta = 0.01`: final `avg50 = 21.46`, `best = 116`
  - `pc-nudge-beta = 0.0001`: final `avg50 = 14.76`, `best = 92`
  - interpretation: the current default remains the best quick probe. Larger LR, more replay updates, and smaller/larger nudge values did not close the gap.
- A critic/policy isolation split was run for seed `42`, 2000 episodes.
  - RTL-faithful Torch tick value + BP policy:
    - command mode: `--value-backend pc --policy-backend bp --pc-gradient-mode pc_nudge_gated_torch_tick --device cuda`
    - result: final `avg50 = 24.14`, `best = 192`
    - interpretation: the tick-faithful critic is the main blocker.
  - BP value + RTL-faithful Torch tick policy:
    - command mode: `--value-backend bp --policy-backend pc --pc-policy-gradient-mode torch_tick --device cuda`
    - result: final `avg50 = 208.98`, `best = 500`
    - interpretation: the PC policy path can support Millidge-style learning when the value model is good.
- A frozen-batch critic sweep was added:
  - script: `scripts/sweep_pc_tick_equivalence.py`
  - output: `python_runs/pc_tick_equivalence_sweep.csv`
  - finding: the residual-error tick critic needs around `200+` inference ticks for good initial gradient direction; default RL runs had been using `--pc-infer 50`.
  - however, transferring the best frozen-batch residual setting to RL did **not** solve the critic.
- Several new critic modes were added to test RTL stories:
  - `pc_nudge_gated_torch_backvec`
    - uses the settled downstream back-vector instead of the hidden residual for the hidden update.
    - initial frozen-batch cosine at 50 ticks improved to about `0.987`.
    - RL critic isolation still failed: final `avg50 = 9.12`, `best = 93`.
  - `pc_nudge_gated_torch_exactlocal`
    - uses tick-settled forward activity, explicit output error, explicit derivative-gated backward vector, and local outer-product gradients.
    - frozen-batch equivalence is essentially exact:
      - cosine `1.000`
      - relative gradient error about `8e-6`
    - RL critic isolation still did not reproduce with default query ticks:
      - final `avg50 = 9.12`, `best = 59`
    - increasing query ticks to `300` helped:
      - `gamma_pc = 0.1`: final `avg50 = 20.60`, `best = 254`
      - `gamma_pc = 0.2`: final `avg50 = 34.84`, `best = 177`
      - `gamma_pc = 0.05` and `0.4` were poor
    - interpretation: even an exact local back-vector critic update is not enough unless the trained tick-query/TD loop remains stable. The remaining gap is now the trained critic query/bootstrapping dynamics, not policy replacement and not GPU execution.
- Critic query-drift diagnostics were added to `python_rtl/run_cartpole_millidge_hybrid.py`.
  - new args:
    - `--critic-drift-every`
    - `--critic-drift-batch`
  - new CSV columns include direct-vs-tick query MSE/max error, direct/tick value magnitudes, target query drift, value weight norm, and target weight gap.
  - failing exact-local critic run without value scaling showed query drift growing badly:
    - query MSE rose from near zero to about `7`
    - max query error rose to about `9`
    - direct/tick value magnitudes rose above `25`
    - target weight gap stayed tiny
  - interpretation: failure was not target-network sync; it was internal PC tick readout leaving the stable scale regime as critic values grew.
- A value-scale fix was tested and worked for critic isolation.
  - command shape:
    - `--pc-gradient-mode pc_nudge_gated_torch_exactlocal --pc-query 300 --gamma-pc 0.2 --pc-value-scale 10 --device cuda`
  - value = PC exact-local, policy = BP, seed `42`, 2000 episodes:
    - final `avg50 = 233.24`
    - final `avg100 = 226.04`
    - `best = 500`
  - query drift stayed much smaller:
    - final query MSE about `0.26`
    - final max query error about `2.71`
  - interpretation: the critic can reproduce when its internal PC activation scale is kept stable. This is the main fix.
- Full RTL-faithful-ish `pc/pc` with the fixed critic was run.
  - command shape:
    - value: `pc_nudge_gated_torch_exactlocal`
    - policy: `torch_tick`
    - `--pc-query 300 --gamma-pc 0.2 --pc-value-scale 10 --device cuda`
  - seed `42`, 2000 episodes:
    - final `avg50 = 166.76`
    - final `avg100 = 162.67`
    - `best = 498`
  - it reached high reward during training:
    - episode 1000 EMA `236.18`
  - interpretation: full PC/PC is now in the high-reward regime, but it is still not a clean solved/reproduced result by final-window criteria. Remaining gap is coupled policy/critic stability, not raw critic failure.
- Coupled policy/critic stability was improved by lowering the PC policy learning rate.
  - tested stabilizers:
    - `--pc-policy-batch-histories`: faster but worse final stability on seed `42` (`avg50 = 119.24`, `best = 500`)
    - `--policy-train-every 10`: unstable final window on seed `42` (`avg50 = 82.14`, `best = 500`)
    - `--lr-policy 0.0003`: best result on seed `42` (`avg50 = 329.38`, `avg100 = 284.31`, `best = 500`)
  - winning full PC/PC command shape:
    - value: `--pc-gradient-mode pc_nudge_gated_torch_exactlocal`
    - policy: `--pc-policy-gradient-mode torch_tick`
    - `--pc-value-query 300`
    - `--pc-policy-query 100`
    - `--gamma-pc 0.2`
    - `--pc-value-scale 10`
    - `--lr-policy 0.0003`
    - `--device cuda`
  - seed `42`, 2000 episodes:
    - final `avg50 = 329.38`
    - final `avg100 = 284.31`
    - `best = 500`
  - confirmation seeds started:
    - seed `1`: final `avg50 = 191.12`, final `avg100 = 234.28`, `best = 500`
    - seed `7`: final `avg50 = 217.62`, final `avg100 = 206.53`, `best = 500`
  - remaining confirmation seeds completed:
    - seed `21`: final `avg50 = 277.84`, final `avg100 = 291.69`, `best = 500`
    - seed `84`: final `avg50 = 280.06`, final `avg100 = 265.37`, `best = 500`
  - five-seed artifacts:
    - `python_runs/rtl_pcpc_exactlocal_scale10_policylr0003_summary_2000.csv`
    - `python_runs/rtl_pcpc_exactlocal_scale10_policylr0003_aggregate_2000.csv`
  - five-seed aggregate over seeds `1, 7, 21, 42, 84`:
    - mean final `avg50 = 259.20`
    - mean final `avg100 = 256.44`
    - mean `best = 500`
    - every seed reached `best = 500`
  - interpretation: full GPU PC/PC reproduction is now achieved by aggregate/final-window criteria. Seed `1` is just below strict `avg50 >= 195` (`191.12`) but clears `avg100` (`234.28`); all other seeds clear both.
- Efficiency improvements were added for future full PC/PC sweeps.
  - `--pc-value-query`: overrides value-network query ticks only.
  - `--pc-policy-query`: overrides policy-network query ticks only.
  - `--pc-policy-batch-histories`: batches all accumulated policy-history states into one Torch tick update instead of stepping one history at a time.
  - Motivation: the fixed critic needs long query settling (`300`), but forcing the policy to use `300` query ticks for every action made long high-reward runs very slow.
  - CUDA smoke tests passed:
    - 50 episodes with `--pc-value-query 300 --pc-policy-query 100 --pc-policy-batch-histories`
    - 200 episodes with the same flags
  - recommended faster full PC/PC command shape:
    - `--pc-gradient-mode pc_nudge_gated_torch_exactlocal`
    - `--pc-policy-gradient-mode torch_tick`
    - `--pc-value-query 300`
    - `--pc-policy-query 100`
    - `--gamma-pc 0.2`
    - `--pc-value-scale 10`
    - `--pc-policy-batch-histories`
    - `--device cuda`

Interpretation: the outer Millidge loop and the PC network as a function approximator are fine. The specific local clamped PC learning rule for the critic is the bottleneck. The next productive direction is to refine/validate `pc_nudge_gated`, then map the required derivative-gating semantics back to the RTL/local-dynamics story.

Likely contributors:

- original full-target clamping moves the hidden state into the wrong gradient regime
- the hidden-layer value update needs the downstream activation derivative gate for BP equivalence
- sequential local PC replay updates may still be weaker/slower than one batched Adam replay update
- the PC critic may need more optimizer-equivalent work per episode to match one backprop replay step

## Roadmap Status

Current answer to "how far along are we?": CartPole is reproduced in the CUDA fast PC bridge, and the first Torch tick-faithful alignment lane exists. The remaining scientific bottleneck is proving that the fast bridge is not hiding a semantic mismatch with local PC tick dynamics.

1. Frozen-Batch Equivalence Report: **done / strong**
   - `scripts/diagnose_pc_bridge_equivalence.py` compares original `pc`, NumPy `pc_nudge_gated`, Torch `pc_nudge_gated_torch_tick`, `pc_nudge_gated_fast`, and `bp_equiv_fast`.
   - Result: original full-clamp PC is not BP-aligned; nudged derivative-gated PC has the right direction but different magnitude; fast bridge is BP-equivalent.

2. Torch Tick-Faithful PC: **full GPU PC/PC CartPole reproduction achieved**
   - Value mode `pc_nudge_gated_torch_tick` and policy mode `torch_tick` are implemented in `python_rtl/run_cartpole_millidge_hybrid.py`.
   - Frozen-batch Torch tick matches NumPy tick closely.
   - 200-episode seed-42 Torch tick is closer to NumPy tick than to the fast bridge.
   - 500-episode five-seed Torch tick is stable but underpowered.
   - BP value + PC policy can reproduce strongly.
   - PC exact-local value + BP policy now reproduces strongly with `--pc-value-scale 10`.
   - Full PC/PC now reaches solved/high-reward final windows with `--lr-policy 0.0003`.
   - Five-seed confirmation is complete.
   - Remaining work: package the result into plots/tables and write the RTL interpretation.

3. Short CartPole Alignment: **done for 50 episodes / needs longer horizon**
   - Seeds `1, 7, 21, 42, 84`, 50 episodes have been run across fast bridge, Torch tick, and NumPy tick.
   - `scripts/exp_cartpole_pc_pc_alignment.py` runs/summarizes the three-way `pc/pc` alignment.
   - Result: fast, Torch tick, and NumPy tick all land around mean `avg50 ~= 21-22`.
   - Longer 200-episode CUDA alignment has been run for fast bridge and Torch tick.
   - Targeted 200-episode NumPy reference for seed `42` is complete.
   - 500-episode Torch tick check is complete and stable, but not solved.
   - Next command should be a structured tuning sweep rather than another one-off run.

4. RTL Story: **not started / next conceptual checkpoint**
   - Current evidence favors an explicit derivative-gated backward vector for the critic update, but the trained query/TD stability problem is not solved.
   - RTL candidates still under consideration:
     - explicit derivative-gated backward vector
     - altered layer activation convention
     - small-nudge clamping phase
     - separate error-gating state in the FSM

5. Pendulum: **PC critic and continuous PC policy seed-42 reproduction complete**
   - New file: `python_rtl/run_pendulum_ddpg.py`
   - This is a standalone CUDA DDPG-style continuous-control runner for `Pendulum-v1`.
   - First seed-42 BP baseline, 150 episodes:
     - CSV: `python_runs/pendulum_ddpg_seed42_150.csv`
     - summary: `python_runs/pendulum_ddpg_summary.csv`
     - final `avg10 = -155.30`
     - final `avg25 = -172.42`
     - best training episode `-0.53`
     - final eval `-118.93`
     - best eval `-41.41`
   - Fast PC critic bridge was added and reproduces the BP critic baseline level.
     - mode: `--critic-backend pc --pc-critic-mode fast --pc-critic-q-scale 10`
     - seed `42`, 150 episodes:
       - final `avg10 = -158.60`
       - final eval `-119.08`
       - best training episode `-0.24`
   - Exact-local PC critic was added.
     - mode: `--critic-backend pc --pc-critic-mode exactlocal`
     - with `--pc-critic-q-scale 10`, it learned early and then collapsed:
       - final `avg10 = -1510.58`
       - final eval `-1558.80`
     - increasing internal Q scaling to `--pc-critic-q-scale 50` fixed the collapse:
       - final `avg10 = -160.68`
       - final eval `-118.48`
       - best training episode `-0.14`
   - Summary artifact:
     - `python_runs/pendulum_pc_critic_summary_seed42_150.csv`
   - Continuous PC actor/policy was added.
     - new args: `--actor-backend bp|pc`, `--pc-actor-mode fast|exactlocal`
     - exact-local actor mode uses the critic action-gradient as the deterministic-policy teaching signal, then applies derivative-gated local layer updates through the actor.
   - Full PC/PC Pendulum exact-local seed-42 CUDA run:
     - mode: `--actor-backend pc --pc-actor-mode exactlocal --critic-backend pc --pc-critic-mode exactlocal --pc-critic-q-scale 50`
     - CSV: `python_runs/pendulum_pcpc_exactlocal_seed42_150.csv`
     - final `avg10 = -156.11`
     - final `avg25 = -184.31`
     - best training episode `-0.24`
     - final eval `-116.14`
     - best eval `-38.31`
   - Full PC/PC Pendulum exact-local five-seed CUDA confirmation is complete.
     - seeds: `1, 7, 21, 42, 84`
     - summary CSV: `python_runs/pendulum_pcpc_exactlocal_multiseed_summary_150.csv`
     - mean final `avg10 = -147.80`
     - std final `avg10 = 9.94`
     - mean final `avg25 = -147.99`
     - mean best training episode `-0.41`
     - mean final eval `-128.83`
     - all five seeds reached near-solved training episodes.
   - Actor/critic replacement comparison for seed 42 is complete.
     - summary CSV: `python_runs/pendulum_actor_critic_comparison_seed42_150.csv`
     - BP actor / BP critic: final `avg10 = -155.30`, final eval `-118.93`
     - BP actor / PC critic exact-local q50: final `avg10 = -160.68`, final eval `-118.48`
     - PC actor exact-local / BP critic: final `avg10 = -167.60`, final eval `-169.69`
     - PC actor exact-local / PC critic exact-local q50: final `avg10 = -156.11`, final eval `-116.14`
   - Robustness sweep for seed 42 is complete.
     - summary CSV: `python_runs/pendulum_pcpc_exactlocal_robustness_seed42_150.csv`
     - `--pc-critic-q-scale 25`, `50`, and `100` all finished in the good band.
     - `--lr-actor 3e-5`, `1e-4`, and `3e-4` all learned, though `3e-4` had weaker final eval.
     - `--pc-query 50` learned early but collapsed by the final window.
     - `--pc-query 100` and `200` stayed stable; use `100` as the default speed/stability point.
   - Interpretation: Pendulum full PC/PC now matches the BP-quality band with exact-local critic and exact-local continuous policy across five seeds. The important fragility is insufficient PC query ticks, not actor replacement or critic scaling within the tested range.

6. MuJoCo: **first full PC/PC GPU run solved InvertedPendulum / multi-seed not stable yet**
   - Installed `gymnasium[mujoco]` dependencies into the conda env.
   - MuJoCo envs verified:
     - `InvertedPendulum-v5`: obs `(4,)`, action `(1,)`, action bound `[-3, 3]`
     - `Reacher-v5`: obs `(10,)`, action `(2,)`
     - `HalfCheetah-v5`: obs `(17,)`, action `(6,)`
   - New file: `python_rtl/run_mujoco_ddpg.py`
     - generalized DDPG runner with dynamic obs/action sizes
     - supports `--actor-backend bp|pc`
     - supports `--critic-backend bp|pc`
     - supports exact-local PC actor and exact-local PC critic on CUDA
   - Smoke tests passed:
     - BP CUDA smoke: `python_runs/mujoco_invertedpendulum_bp_smoke.csv`
     - PC/PC exact-local CUDA update smoke: `python_runs/mujoco_invertedpendulum_pcpc_smoke.csv`
   - First corrected BP baseline:
     - env: `InvertedPendulum-v5`
     - seed `42`, 300 episodes, early updates with `--start-steps 0 --update-after 64 --batch-size 64`
     - CSV: `python_runs/mujoco_invertedpendulum_bp_seed42_300_earlyupdate.csv`
     - best training episode `1000`
     - best eval `1000`
     - final `avg10 = 714.3`
     - final eval collapsed to `6.0`
     - interpretation: minimal DDPG can solve but is not tail-stable yet.
   - BP baseline tuning/stability checks:
     - `python_rtl/run_mujoco_ddpg.py` now supports `--exploration-noise-final`, `--exploration-noise-decay-episodes`, `--freeze-actor-after-eval`, and CSV best-eval tracking.
     - It also now supports in-memory best-actor checkpointing and end-of-run final/best-checkpoint evaluation with `--final-eval-episodes`.
     - summary CSV: `python_runs/mujoco_invertedpendulum_bp_tuning_pcpc_retention_seed42.csv`
     - noise decay alone did not solve:
       - CSV: `python_runs/mujoco_invertedpendulum_bp_seed42_300_noise_decay.csv`
       - final eval `37.33`, best eval `47.33`
     - delayed random-start replay did not solve:
       - CSV: `python_runs/mujoco_invertedpendulum_bp_seed42_400_start1000.csv`
       - final eval `59.0`, best eval `62.0`
     - solved-eval actor freeze stabilized the BP baseline after discovery:
       - CSV: `python_runs/mujoco_invertedpendulum_bp_seed42_300_freeze950.csv`
       - final eval `1000.0`, best eval `1000.0`
       - final training `avg10 = 335.1` because exploration noise remains during training episodes.
     - interpretation: BP already had a solved policy but kept drifting; freezing actor updates after solved eval cleanly separates policy discovery from post-discovery drift.
   - First full PC/PC exact-local MuJoCo run:
     - env: `InvertedPendulum-v5`
     - seed `42`, 300 episodes
     - mode: `--actor-backend pc --pc-actor-mode exactlocal --critic-backend pc --pc-critic-mode exactlocal --pc-critic-q-scale 100 --pc-query 100`
     - early updates: `--start-steps 0 --update-after 64 --batch-size 64`
     - CSV: `python_runs/mujoco_invertedpendulum_pcpc_seed42_300_earlyupdate.csv`
     - final `avg10 = 1000.0`
     - final eval `1000.0`
     - best training episode `1000.0`
     - best eval `1000.0`
   - PC/PC with the same solved-eval actor-freeze scaffold:
     - CSV: `python_runs/mujoco_invertedpendulum_pcpc_seed42_300_freeze950.csv`
     - final `avg10 = 1000.0`
     - final eval `1000.0`
     - best eval `1000.0`
   - Comparison artifact:
     - `python_runs/mujoco_invertedpendulum_seed42_comparison_300.csv`
   - Multi-seed InvertedPendulum PC/PC confirmation has been run for seeds `1, 7, 21, 42, 84`.
     - summary CSV: `python_runs/mujoco_invertedpendulum_pcpc_multiseed_summary_300.csv`
     - seed `42`: final `avg10 = 1000.0`, final eval `1000.0`, solved final window.
     - seeds `7` and `21`: each reached at least one `1000` training episode but did not retain solved behavior.
     - seeds `1` and `84`: improved to medium control but did not solve within 300 episodes.
     - mean final `avg10 = 269.2`, heavily skewed by seed `42`.
   - Best-checkpoint multi-seed comparison has been run for BP and PC/PC.
     - combined CSV: `python_runs/mujoco_invertedpendulum_bp_vs_pcpc_bestckpt_multiseed_300.csv`
     - BP summary CSV: `python_runs/mujoco_invertedpendulum_bp_bestckpt_multiseed_summary_300.csv`
     - PC/PC summary CSV: `python_runs/mujoco_invertedpendulum_pcpc_bestckpt_multiseed_summary_300.csv`
     - BP best-eval solved rate: `1/5`
     - BP final-eval solved rate: `0/5`
     - PC/PC best-eval solved rate: `1/5`
     - PC/PC final-eval solved rate: `1/5`
     - BP mean best eval: `255.87`
     - PC/PC mean best eval: `288.47`
     - interpretation: best-checkpoint evaluation confirms the current minimal DDPG scaffolding is still the bottleneck; both BP and PC/PC solve seed `42`, while other seeds mostly reach medium control or transient high training episodes without deterministic solved eval.
   - Targeted stabilization attempts on seed `7`:
     - lower actor LR `--lr-actor 0.00003`: final `avg10 = 59.9`, best `263.0`
     - higher query count `--pc-query 200`: final `avg10 = 92.9`, best `155.0`
     - neither fixed solved-policy retention.
   - First multi-action MuJoCo test on `Reacher-v5` is complete.
     - PC/PC CUDA smoke passed with obs dim `10`, action dim `2`.
     - BP seed-42 100 episodes:
       - CSV: `python_runs/mujoco_reacher_bp_seed42_100.csv`
       - final `avg10 = -8.42`, final eval `-7.57`, best `-2.90`
     - PC/PC exact-local seed-42 100 episodes:
       - CSV: `python_runs/mujoco_reacher_pcpc_seed42_100.csv`
       - final `avg10 = -12.76`, final eval `-12.28`, best `-8.62`
     - comparison CSV: `python_runs/mujoco_reacher_seed42_comparison_100.csv`
   - Interpretation: the PC actor/critic machinery generalizes to MuJoCo and vector actions, but the current minimal DDPG setup is not yet a robust MuJoCo benchmark story. InvertedPendulum needs solved-policy retention across seeds, and Reacher needs tuning/longer runs to match BP.
   - Conventional TD3 baseline work has started.
     - New homebrew TD3 runner: `python_rtl/run_mujoco_td3.py`
       - twin BP critics
       - clipped min target Q
       - delayed policy updates
       - target policy smoothing
       - best-checkpoint evaluation
     - Homebrew TD3 fixed the seed-42 warmup issue when run with:
       - `--episodes 600 --start-steps 1000 --update-after 1000 --batch-size 100 --lr-actor 0.0003 --lr-critic 0.0003`
       - CSV: `python_runs/mujoco_td3_invertedpendulum_seed42_600_start1000_lr3e-4.csv`
       - final avg10 `993.7`
       - final eval `1000.0`
       - best checkpoint eval `1000.0`
     - Homebrew TD3 still failed seed `1` with that config:
       - CSV: `python_runs/mujoco_td3_invertedpendulum_seed1_600_start1000_lr3e-4.csv`
       - final eval `45.3`
     - Homebrew TD3 was then corrected to match the SB3 timing/noise convention more closely:
       - `python_rtl/run_mujoco_td3.py` now supports `--total-timesteps` and `--eval-every-steps`.
       - Exploration noise and target-policy smoothing noise now default to SB3-style absolute action units.
       - The old range-relative noise behavior is still available with `--scale-action-noise`.
       - Root cause of the earlier seed-1 failure: 600 short episodes only gave about `13k` environment steps, far below the SB3 100k-step reference budget, and the old action-range-scaled noise was too aggressive for `InvertedPendulum-v5`.
     - Corrected homebrew BP TD3 100k-step five-seed run:
       - shared command shape:
         - `--total-timesteps 100000 --start-steps 1000 --update-after 1000 --batch-size 256 --lr-actor 0.0003 --lr-critic 0.0003 --exploration-noise 0.1 --policy-noise 0.2 --noise-clip 0.5 --eval-every-steps 5000`
       - seed `1`: CSV `python_runs/mujoco_td3_bp_noisefix_seed1_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
       - seed `7`: CSV `python_runs/mujoco_td3_bp_noisefix_seed7_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
       - seed `21`: CSV `python_runs/mujoco_td3_bp_noisefix_seed21_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
       - seed `42`: CSV `python_runs/mujoco_td3_bp_noisefix_seed42_100k.csv`, final avg10 `961.3`, final eval `104.0`, best checkpoint eval `1000.0`
       - seed `84`: CSV `python_runs/mujoco_td3_bp_noisefix_seed84_100k.csv`, final avg10 `937.7`, final eval `1000.0`, best eval `1000.0`
       - interpretation: the editable homebrew BP TD3 scaffold now reaches solved checkpoints on all five reference seeds at the same 100k-step budget as SB3. Final-policy retention is still imperfect on seed `42`, so best-checkpoint selection or slightly gentler late actor updates remain useful before porting the same sweep to BP-AIF and PC/PC.
     - Stable-Baselines3 TD3 runner added: `python_rtl/run_mujoco_sb3_td3.py`
       - This is the conventional RL reference path for MuJoCo.
       - Seed `42`, `InvertedPendulum-v5`, 100k timesteps:
         - CSV: `python_runs/mujoco_sb3_td3_invertedpendulum_seed42_100k.csv`
         - solved by 25k timesteps
         - held eval `1000.0` through 100k
         - final eval `1000.0`
       - Five-seed reference is complete for seeds `1, 7, 21, 42, 84`.
         - summary CSV: `python_runs/mujoco_sb3_td3_invertedpendulum_multiseed_summary_100k.csv`
         - final solved rate `5/5`
         - best solved rate `5/5`
         - mean final eval `1000.0`
         - mean best eval `1000.0`
         - first solved step:
           - seed `1`: 30k
           - seed `7`: 30k
           - seed `21`: 30k
           - seed `42`: 25k
           - seed `84`: 20k
     - Interpretation: MuJoCo is not the problem; the earlier instability came from the toy DDPG/homebrew scaffolding. The serious path is TD3-style scaffolding first, then PC critic/actor replacement inside that framework.
     - Connection to Millidge / BP Deep AIF:
       - Define the AIF critic as expected free energy with `G(s,a) = -Q(s,a)`.
       - TD3 critic learning becomes learning a stabilized expected-free-energy critic under the sign flip.
       - TD3 actor update `maximize Q(s, pi(s))` becomes AIF actor update `minimize G(s, pi(s))`.
       - Therefore the first BP-AIF target is to reproduce the SB3 TD3 curve/functionality under this EFE-sign convention before replacing BP with exact-local PC updates.
     - BP-AIF TD3 conversion is implemented in `python_rtl/run_mujoco_td3.py`.
       - New arg: `--critic-semantics q|aif`
       - `--critic-semantics aif` uses `AmortizedEFECritic`, whose forward output is `G(s,a) = -raw_network(s,a)`.
       - TD3 target in AIF mode is `target_G = -reward + gamma * max(G1_target, G2_target)`.
       - Actor loss in AIF mode is `mean(G(s, pi(s)))`.
       - This preserves exact BP dynamics relative to TD3 Q-learning while exposing the Millidge amortized EFE semantics.
     - Seed-42 conversion check:
       - comparison CSV: `python_runs/mujoco_td3_aif_conversion_seed42_comparison.csv`
       - Q-mode TD3: final avg10 `993.7`, final eval `1000.0`, first solved eval episode `550`
       - signed BP-AIF TD3: final avg10 `993.7`, final eval `1000.0`, first solved eval episode `550`
       - direct randomly initialized G-mode failed: final avg10 `50.9`, final eval `53.2`
       - interpretation: for BP equivalence, the amortized EFE head must maintain the `G=-Q` parameterization/sign convention, not merely relabel a randomly initialized critic as G.
     - Corrected 100k-step Millidge BP-AIF TD3 five-seed sweep is complete.
       - shared command shape:
         - `--total-timesteps 100000 --critic-semantics aif --critic-backend bp --actor-backend bp --start-steps 1000 --update-after 1000 --batch-size 256 --lr-actor 0.0003 --lr-critic 0.0003 --exploration-noise 0.1 --policy-noise 0.2 --noise-clip 0.5 --eval-every-steps 5000`
       - seed `1`: CSV `python_runs/mujoco_td3_aif_bp_seed1_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
       - seed `7`: CSV `python_runs/mujoco_td3_aif_bp_seed7_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
       - seed `21`: CSV `python_runs/mujoco_td3_aif_bp_seed21_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
       - seed `42`: CSV `python_runs/mujoco_td3_aif_bp_seed42_100k.csv`, final avg10 `961.3`, final eval `104.0`, best checkpoint eval `1000.0`
       - seed `84`: CSV `python_runs/mujoco_td3_aif_bp_seed84_100k.csv`, final avg10 `937.7`, final eval `1000.0`, best eval `1000.0`
       - interpretation: the corrected Millidge BP-AIF implementation now transfers the fixed homebrew TD3 result exactly at the functional level. It reaches solved checkpoints on all five seeds under `G = -Q`; four seeds retain solved final policies, while seed `42` has the same late-drift pattern as Q-mode TD3 and is recovered by best-checkpoint evaluation.
     - PC/PC replacement inside the signed TD3-AIF scaffold is implemented in `python_rtl/run_mujoco_td3.py`.
       - New args: `--actor-backend bp|pc`, `--critic-backend bp|pc`, `--pc-query`, `--pc-critic-value-scale`.
       - PC actor: exact-local deterministic actor update from the critic-supplied action teaching signal.
       - PC critic: exact-local tick-settled critic update with signed AIF semantics; in AIF mode the public critic output is `G = -raw_Q`, and the PC local target is internally sign-flipped.
       - Full PC/PC CUDA smoke passed:
         - CSV: `python_runs/mujoco_td3_aif_pcpc_smoke.csv`
       - Staged PC critic / BP actor seed-42 300-episode run:
         - CSV: `python_runs/mujoco_td3_aif_pccritic_bpa_seed42_300.csv`
         - final avg10 `77.9`
         - final eval `106.8`
       - Full PC/PC signed TD3-AIF seed-42 300-episode run:
         - CSV: `python_runs/mujoco_td3_aif_pcpc_seed42_300.csv`
         - final avg10 `109.2`
         - best eval `116.4`
     - Full PC/PC signed TD3-AIF seed-42 600-episode run:
       - CSV: `python_runs/mujoco_td3_aif_pcpc_seed42_600.csv`
       - solved checkpoint at episode `500`
       - best eval `1000.0`
       - best checkpoint eval `1000.0`
       - final avg10 `20.9`
       - final policy eval `28.0`
     - Interpretation: the PC/PC replacement is now functionally working and can hit the same solved InvertedPendulum behavior as the BP-AIF conversion on seed `42`, but it does not yet retain the solved policy through the final window. The remaining issue is stability/retention under coupled PC actor plus PC twin critics, not whether the Millidge/Td3 `G=-Q` replacement can learn at all.
     - Corrected 100k-step PC/PC transfer work has started.
       - First full PC/PC 100k-step seed-42 attempt with `--pc-critic-value-scale 100` showed early critic alignment near 1.0 but degraded around episode `300`, with reward collapse. This identified target normalization/settling scale as the first PC bottleneck.
       - Diagnostic fix: `PCTickCritic.measure_bp_cosine` now computes the BP comparison gradient correctly instead of running under `no_grad`.
       - BP-equivalent PC critic update mode was added:
         - new arg: `--pc-critic-gradient-mode exactlocal|bp_equiv`
         - `bp_equiv` keeps the PC critic substrate/class but applies the exact BP gradient to that PC network.
         - This separates the substrate replacement claim from the tick-local RTL approximation claim.
       - Full PC/PC BP-equivalent AIF TD3 100k-step five-seed sweep is complete.
         - shared command shape:
           - `--total-timesteps 100000 --critic-semantics aif --critic-backend pc --actor-backend pc --pc-critic-gradient-mode bp_equiv --pc-critic-value-scale 1 --start-steps 1000 --update-after 1000 --batch-size 256 --lr-actor 0.0003 --lr-critic 0.0003 --exploration-noise 0.1 --policy-noise 0.2 --noise-clip 0.5 --eval-every-steps 5000`
         - seed `1`: CSV `python_runs/mujoco_td3_aif_pcpc_bpequiv_seed1_100k.csv`, final avg10 `888.9`, final eval `1000.0`, best eval `1000.0`
         - seed `7`: CSV `python_runs/mujoco_td3_aif_pcpc_bpequiv_seed7_100k.csv`, final avg10 `877.7`, final eval `1000.0`, best eval `1000.0`
         - seed `21`: CSV `python_runs/mujoco_td3_aif_pcpc_bpequiv_seed21_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
         - seed `42`: CSV `python_runs/mujoco_td3_aif_pcpc_bpequiv_seed42_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
         - seed `84`: CSV `python_runs/mujoco_td3_aif_pcpc_bpequiv_seed84_100k.csv`, final avg10 `1000.0`, final eval `1000.0`, best eval `1000.0`
         - interpretation: full PC/PC AIF replication is now achieved by BP-equivalent PC update criteria. The PC actor and PC critic substrates can replace BP networks while preserving the corrected Millidge BP-AIF TD3 result across all five reference seeds.
       - Better 30k-step seed-42 PC critic / BP actor probe:
         - CSV: `python_runs/mujoco_td3_aif_pccritic_bpa_seed42_30k_lr1e-4_q200_scale1000.csv`
         - command knobs: `--critic-backend pc --actor-backend bp --lr-critic 0.0001 --pc-query 200 --pc-critic-value-scale 1000`
         - best eval `294.8`
         - final eval `283.2`
         - final avg10 `167.5`
         - critic grad cosine stayed high through late diagnostics (`~0.998` by episode `500`)
       - Better 30k-step seed-42 full PC/PC probe:
         - CSV: `python_runs/mujoco_td3_aif_pcpc_seed42_30k_lr1e-4_q200_scale1000.csv`
         - command knobs: `--critic-backend pc --actor-backend pc --lr-actor 0.0001 --lr-critic 0.0001 --pc-query 200 --pc-critic-value-scale 1000`
         - best eval `195.2`
         - final eval `198.0`
         - final avg10 `166.6`
         - critic grad cosine stayed near-perfect through late diagnostics (`~0.999998` at episode `500`)
       - Interpretation: `pc-critic-value-scale 1000` fixes the critic-alignment collapse seen with scale `100`. The replacement is now stable and learning under the corrected TD3 timing/noise budget, but it has not yet reproduced solved BP-AIF behavior at 30k steps. Next run should extend the scale-1000 config toward 100k steps and/or tune learning speed.

## Current Files To Use

If continuing this project, start from:

- `python_rtl/run_cartpole_millidge_bp.py`
- `python_rtl/run_cartpole_millidge_hybrid.py`

Use these modes in the hybrid runner:

- baseline reference: `--value-backend bp --policy-backend bp`
- first replacement: `--value-backend pc --policy-backend bp`
- diagnostic bridge: `--value-backend pc --policy-backend bp --pc-gradient-mode bp_equiv`
- current best PC-style critic update: `--value-backend pc --policy-backend bp --pc-gradient-mode pc_nudge_gated --pc-nudge-beta 0.001 --pc-infer 300 --max-infer-ticks 300 --no-adaptive-inference`
- current long-sweep CUDA bridge: `--value-backend pc --policy-backend bp --pc-gradient-mode pc_nudge_gated_fast --device cuda`
- current full `pc/pc` CUDA bridge: `--value-backend pc --policy-backend pc --pc-gradient-mode pc_nudge_gated_fast --pc-policy-gradient-mode fast --device cuda`
- clean CartPole comparison report: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_cartpole_pc_bridge.py --run-missing --device cuda`
- bridge equivalence diagnostic: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/diagnose_pc_bridge_equivalence.py --run-cartpole --cartpole-episodes 50 --cartpole-seeds 42`
- Torch tick `pc/pc` alignment command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python python_rtl/run_cartpole_millidge_hybrid.py --episodes 50 --seed 42 --value-backend pc --policy-backend pc --pc-gradient-mode pc_nudge_gated_torch_tick --pc-policy-gradient-mode torch_tick --device cuda`
- short `pc/pc` alignment report: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_cartpole_pc_pc_alignment.py --run-missing --episodes 50 --seeds 1 7 21 42 84`
- selected-mode alignment report: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_cartpole_pc_pc_alignment.py --run-missing --episodes 200 --seeds 1 7 21 42 84 --modes pc_pc_fast pc_pc_torch_tick`
- tagged seed-42 NumPy spot check: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_cartpole_pc_pc_alignment.py --run-missing --episodes 200 --seeds 42 --modes pc_pc_numpy_tick --tag seed42_spotcheck`

## Suggested Next Steps

1. Package the CartPole full PC/PC five-seed result into plots/tables and write the RTL interpretation.
2. Package the Pendulum full PC/PC five-seed result into plots/tables.
3. Write the continuous-action RTL interpretation for the PC actor:
   - critic supplies the local action-error teaching signal
   - actor applies derivative-gated local updates through tanh and ReLU layers
   - query ticks must settle enough to avoid late critic/policy drift
4. Stabilize MuJoCo InvertedPendulum across seeds. Likely next knobs:
   - Use `python_rtl/run_mujoco_sb3_td3.py` as the conventional reference.
   - Homebrew BP TD3 now reaches solved checkpoints on all five reference seeds at 100k timesteps.
   - BP-AIF TD3 equivalence is implemented and five-seed verified at 100k timesteps under the `G = -Q` convention.
   - Full PC/PC BP-equivalent AIF TD3 is five-seed verified at 100k timesteps.
   - Corrected timing/noise budget has been ported to PC/PC. Current best stable seed-42 PC/PC probe uses `--lr-actor 0.0001 --lr-critic 0.0001 --pc-query 200 --pc-critic-value-scale 1000`.
   - Next step: close the gap between BP-equivalent PC and exact-local/tick PC by extending the stable exact-local config to 100k steps, then tune learning speed/retention: checkpoint early-stop criteria, actor LR, critic value-scale/query sweep, and then multi-seed confirmation.
5. Reacher-v5 TD3/AIF/PC-PC transfer has started in a separate script.
   - New script: `scripts/exp_reacher_pcpc_aif_td3.py`
   - It runs the same ladder as InvertedPendulum:
     - `bp_q`: conventional homebrew TD3
     - `bp_aif`: Millidge-style BP-AIF TD3 with `G = -Q`
     - `pcpc_aif_bpequiv`: PC actor + PC critic with BP-equivalent PC critic updates
   - CUDA smoke passed for seed `42`, 1k timesteps:
     - obs dim `10`, action dim `2`
     - all three lanes matched on the smoke run
   - Seed-42 100k-step Reacher run completed:
     - summary CSV: `python_runs/reacher_td3_pcpc_summary_100000.csv`
     - aggregate CSV: `python_runs/reacher_td3_pcpc_aggregate_100000.csv`
     - BP TD3 Q: final eval `-9.002`, best eval `-7.028`
     - BP-AIF `G=-Q`: final eval `-9.002`, best eval `-7.028`
     - PC/PC AIF BP-equivalent: final eval `-7.942`, best eval `-7.319`
   - Five-seed 100k-step Reacher confirmation is complete for seeds `1, 7, 21, 42, 84`.
     - summary CSV: `python_runs/reacher_td3_pcpc_summary_100000.csv`
     - aggregate CSV: `python_runs/reacher_td3_pcpc_aggregate_100000.csv`
     - BP TD3 Q:
       - mean final avg50 `-11.666`
       - mean final eval `-8.984`
       - mean best eval `-6.364`
     - BP-AIF `G=-Q`:
       - mean final avg50 `-11.666`
       - mean final eval `-8.984`
       - mean best eval `-6.364`
       - matches BP TD3 Q exactly at aggregate level
     - PC/PC AIF BP-equivalent:
       - mean final avg50 `-11.088`
       - mean final eval `-9.579`
       - mean best eval `-6.838`
   - Interpretation: BP-AIF reproduces the TD3 Q lane exactly on Reacher, and the PC/PC BP-equivalent substrate lands in the same performance band across five seeds. This confirms the InvertedPendulum transfer path generalizes to a vector-action MuJoCo task. The baseline itself is modest/noisy on Reacher, so a stronger Reacher story would tune the TD3 baseline further, but the PC/PC replacement claim is supported.
6. HalfCheetah-v5 TD3/AIF/PC-PC transfer has started in a separate script.
   - New script: `scripts/exp_halfcheetah_pcpc_aif_td3.py`
   - It runs the same ladder:
     - `bp_q`: conventional homebrew TD3
     - `bp_aif`: Millidge-style BP-AIF TD3 with `G = -Q`
     - `pcpc_aif_bpequiv`: PC actor + PC critic with BP-equivalent PC critic updates
   - CUDA smoke passed for seed `42`, 1k timesteps:
     - obs dim `17`, action dim `6`
     - BP-AIF matched BP-Q immediately
     - PC/PC vector-action lane ran successfully
   - Seed-42 100k-step HalfCheetah run completed:
     - summary CSV: `python_runs/halfcheetah_td3_pcpc_summary_100000.csv`
     - aggregate CSV: `python_runs/halfcheetah_td3_pcpc_aggregate_100000.csv`
     - BP TD3 Q:
       - final eval `3990.616`
       - best checkpoint eval `4096.804`
       - final avg10 `3369.659`
     - BP-AIF `G=-Q`:
       - final eval `3990.616`
       - best checkpoint eval `4096.804`
       - final avg10 `3369.659`
       - matches BP TD3 Q exactly
     - PC/PC AIF BP-equivalent:
       - final eval `3687.905`
       - best checkpoint eval `3719.560`
       - final avg10 `3522.072`
   - Five-seed 100k-step HalfCheetah confirmation is complete for seeds `1, 7, 21, 42, 84`.
     - summary CSV: `python_runs/halfcheetah_td3_pcpc_summary_100000.csv`
     - aggregate CSV: `python_runs/halfcheetah_td3_pcpc_aggregate_100000.csv`
     - BP TD3 Q:
       - mean final avg50 `3113.118`
       - mean final eval `3796.312`
       - mean best eval `4012.140`
     - BP-AIF `G=-Q`:
       - mean final avg50 `3113.118`
       - mean final eval `3796.312`
       - mean best eval `4012.140`
       - matches BP TD3 Q exactly at aggregate level
     - PC/PC AIF BP-equivalent:
       - mean final avg50 `2856.203`
       - mean final eval `3804.795`
       - mean best eval `3804.795`
   - Interpretation: BP-AIF reproduces TD3 Q exactly on HalfCheetah, and PC/PC BP-equivalent reaches the same strong locomotion regime across five seeds. PC/PC mean final eval is essentially matched/slightly higher, while mean best eval is lower than BP. This is the strongest high-dimensional MuJoCo locomotion transfer so far.

## Hardware / GPU Note

Post-reboot GPU status:

- `nvidia-smi` works outside the sandbox
- loaded NVIDIA kernel module: `580.159.03`
- conda PyTorch: `2.6.0+cu124`
- `torch.cuda.is_available()` is `True`
- detected GPU: `NVIDIA GeForce RTX 4090`
- CUDA matmul smoke test passed
- hybrid runner CUDA smoke test passed with `using device: cuda`

Important environment note: the Codex sandbox still hides `/dev/nvidia*`, so GPU commands need to run outside the sandbox / with escalation. Use the conda Python explicitly:

`/home/gregoryv/miniconda3/envs/dsl2/bin/python python_rtl/run_cartpole_millidge_hybrid.py --device cuda --print-device ...`

CartPole, Pendulum, and MuJoCo now all have CUDA-capable PC experiment paths. Some legacy NumPy tick-faithful comparison modes are still CPU-bound, but the current Torch/exact-local runs use the GPU when launched with `--device cuda`.

## RTL-Faithful Verification Start

Current scientific bottleneck: move from functional PC/PC BP-equivalent AIF replication to RTL-faithful physical inference-engine verification.

First checks completed:

- Read `mypaper.pdf` and compared its local PC equations against `rtl/nc_neuralcomputer.sv`.
- The RTL neuron FSM implements the paper dynamics:
  - `PRED`: `mu_i = sum_j theta_ij * phi(x_j_up) + bias`
  - `ERR`: `eps_i = x_eff - mu_i`
  - `BACKSUM/BACKVEC`: `back_eff = phi'(x_eff) * sum(back_in)`, `back_vec[j] = theta_j * eps_i`
  - `WUP`: `theta_j += alpha * eps_i * phi(x_j_up)`
  - `STATE`: `x_i += gamma * (back_eff - eps_i)` unless hard-clamped
- Added primitive RTL trace testbench: `tb/tb_neuron_tick_trace.sv`
- Added Python oracle checker: `scripts/check_neuron_tick_trace.py`
- Added primitive sweep runner: `scripts/exp_rtl_primitive_sweep.py`
- Verification result:
  - command: `./scripts/run_test.sh tb/tb_neuron_tick_trace.sv tb_neuron_tick_trace -- +CSV=runs/neuron_tick_trace.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/check_neuron_tick_trace.py --csv runs/neuron_tick_trace.csv`
  - result: `PASS neuron tick RTL matches Python oracle`
  - max absolute error: `4.91172797e-10`
- Primitive sweep result:
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_primitive_sweep.py`
  - summary: `runs/rtl_primitive_sweep_summary.csv`
  - cases passed:
    - `linear_free`: max abs `4.91172797e-10`
    - `linear_soft_xset`: max abs `2.94209747e-08`
    - `linear_hard_xset`: max abs `2.94209747e-08`
    - `relu_free`: max abs `4.9011612e-10`
    - `relu_soft_xset`: max abs `4.70348357e-10`
    - `relu_hard_xset`: max abs `4.70348357e-10`
- Added `PCNet3Layer.tick_parallel()` in `python_rtl/pc_network.py` to snapshot inter-layer state/backflow before layer commits. This matches RTL `pc_network_nlayer` simultaneous `start_tick` semantics better than the old sequential/Gauss-Seidel Python helper.
- Added new general Python reference class `PCNetNLayer` in `python_rtl/pc_network.py`.
  - It is separate from `PCNet3Layer`; existing 3-layer dynamics are preserved.
  - It supports arbitrary bottom-to-top `k_lut` / `act_lut` stacks.
  - `tick_parallel()` uses simultaneous RTL-boundary timing for all layers.
  - It defaults to RTL-like top-layer presynaptic width (`top_rtl_width=True`) for network trace work; `top_rtl_width=False` can be used to match legacy `PCNet3Layer` behavior exactly.
  - Important trace detail: clamped layers expose their observed/effective state at the RTL boundary, so `PCNetNLayer.tick_parallel()` uses `x_eff = obs` for clamped upstream layers when building the next layer's `x_up`.
- Added `scripts/check_pcnet_nlayer.py`.
  - It verifies `PCNetNLayer(..., top_rtl_width=False)` matches `PCNet3Layer.tick_parallel()` exactly on a 3-layer smoke.
  - It also runs a 4-layer smoke to confirm the generalized class functions beyond three layers.
  - Result: `PASS PCNetNLayer checks`
- Added deterministic RTL network trace:
  - testbench: `tb/tb_network_tick_trace.sv`
  - checker: `scripts/check_network_tick_trace.py`
  - output: `runs/network_tick_trace.csv`
  - command: `./scripts/run_test.sh tb/tb_network_tick_trace.sv tb_network_tick_trace -- +CSV=runs/network_tick_trace.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/check_network_tick_trace.py --csv runs/network_tick_trace.csv`
  - result: `PASS network tick RTL matches PCNetNLayer`
  - max absolute error: `2.4e-08`
  - trace compares layer states plus representative weights/biases/back-vector fields across 4 deterministic ticks.
- Parameterized `tb/tb_network_tick_trace.sv` over activation IDs and runtime `+ALPHA` / `+GAMMA`.
- Added network trace sweep:
  - script: `scripts/exp_rtl_network_trace_sweep.py`
  - summary: `runs/rtl_network_trace_sweep_summary.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_network_trace_sweep.py`
  - cases passed:
    - `linear/relu/linear`, alpha `0.05`, gamma `0.10`
    - `linear/relu/linear`, alpha `0.02`, gamma `0.05`
    - `linear/relu/linear`, alpha `0.10`, gamma `0.20`
    - `linear/linear/linear`, alpha `0.05`, gamma `0.10`
    - `relu/relu/linear`, alpha `0.05`, gamma `0.10`
  - all cases max absolute error: `2.4e-08`
  - Note: tanh/sigmoid RTL blocks are approximate hardware activation paths, so they should get an approximation-aware oracle before being included in exact trace sweeps.
- Added `--schedule {sequential,parallel}` and `--learn-gamma-mode {frozen,rtl}` to `python_rtl/tb_scale_function.py`.
  - Important mismatch found: the RTL scale testbench keeps `gamma` active during learning ticks, while the historical Python scale path froze state updates with `gamma=0` during learn ticks.
- Patched `tb/tb_scale_function.sv` to seed via Verilator-supported `$urandom(RAND_SEED)` instead of unsupported `$srandom`.
- RTL supervised smoke now runs:
  - command: `./scripts/run_test.sh tb/tb_scale_function.sv tb_scale_function -- +CSV=runs/scale_2_4_3_rtl_smoke.csv +N_SAMPLES=8 +SEED=0 +EPOCHS=2 +INFER_TICKS=20 +LEARN_TICKS=3 +EVAL_TICKS=40 +ALPHA=0.05 +GAMMA=0.10`
  - MSE: `0.139199 -> 0.109118 -> 0.103912`
- Python supervised smoke with RTL-aligned schedule also runs:
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python python_rtl/tb_scale_function.py --configs 2_4_3 --teacher tiled --act-hidden relu --alpha 0.05 --gamma 0.10 --infer-ticks 20 --learn-ticks 3 --eval-settle 40 --epochs 2 --n-samples 8 --seed 0 --schedule parallel --learn-gamma-mode rtl`
  - MSE: `0.828441 -> 0.078810 -> 0.075311`
- Added deterministic supervised RTL-vs-Python learning trace:
  - testbench: `tb/tb_supervised_fixed_trace.sv`
  - checker: `scripts/check_supervised_fixed_trace.py`
  - output: `runs/supervised_fixed_trace.csv`
  - command: `./scripts/run_test.sh tb/tb_supervised_fixed_trace.sv tb_supervised_fixed_trace -- +CSV=runs/supervised_fixed_trace.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/check_supervised_fixed_trace.py --csv runs/supervised_fixed_trace.csv`
  - result: `PASS supervised fixed RTL learning curve matches PCNetNLayer`
  - max absolute MSE error: `4.75322975e-10`
  - RTL MSE curve: `0.038752304 -> 0.040932202 -> 0.040604896 -> 0.040416668`
  - This uses a fixed 4-sample dataset, no RNG, 2->4->3 linear/ReLU/linear network, RTL-style parallel tick schedule, gamma active during learning.
- Parameterized `tb/tb_supervised_fixed_trace.sv` and `scripts/check_supervised_fixed_trace.py` over epochs, infer ticks, learn ticks, eval ticks, alpha, and gamma.
- Added supervised fixed-dataset sweep:
  - script: `scripts/exp_rtl_supervised_fixed_sweep.py`
  - summary: `runs/rtl_supervised_fixed_sweep_summary.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_supervised_fixed_sweep.py`
  - cases passed:
    - base: epochs `3`, infer `10`, learn `2`, eval `20`, alpha `0.05`, gamma `0.10`, max abs `4.75322975e-10`, final MSE `0.040416668`
    - short: epochs `3`, infer `5`, learn `1`, eval `10`, alpha `0.02`, gamma `0.05`, max abs `1.16381301e-09`, final MSE `0.058083781`
    - strong: epochs `3`, infer `10`, learn `2`, eval `20`, alpha `0.10`, gamma `0.20`, max abs `4.33792217e-10`, final MSE `0.037921148`
    - longer: epochs `5`, infer `15`, learn `3`, eval `25`, alpha `0.05`, gamma `0.10`, max abs `6.53811304e-10`, final MSE `0.039032147`
- Added larger deterministic supervised grid trace:
  - testbench: `tb/tb_supervised_grid_trace.sv`
  - checker: `scripts/check_supervised_grid_trace.py`
  - smoke output: `runs/supervised_grid_2_4_3.csv`
  - command: `./scripts/run_test.sh tb/tb_supervised_grid_trace.sv tb_supervised_grid_trace -GK0=3 -GK1=4 -GK2=2 -GNUM_SAMPLES=8 -- +CSV=runs/supervised_grid_2_4_3.csv +EPOCHS=2 +INFER_TICKS=8 +LEARN_TICKS=2 +EVAL_TICKS=12 +ALPHA=0.05 +GAMMA=0.10`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/check_supervised_grid_trace.py --csv runs/supervised_grid_2_4_3.csv --k0 3 --k1 4 --k2 2 --samples 8 --epochs 2 --infer-ticks 8 --learn-ticks 2 --eval-ticks 12 --alpha 0.05 --gamma 0.10`
  - result: `PASS supervised grid RTL learning curve matches PCNetNLayer`
  - max absolute MSE error: `2.65814719e-09`
- Added larger supervised grid/config sweep:
  - script: `scripts/exp_rtl_supervised_grid_sweep.py`
  - summary: `runs/rtl_supervised_grid_sweep_summary.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_supervised_grid_sweep.py`
  - cases passed:
    - `3->4->2`, samples `8`, epochs `2`, max abs `2.65814719e-09`, final MSE `0.082509124`
    - `4->8->4`, samples `12`, epochs `2`, max abs `3.09789143e-09`, final MSE `0.093101249`
    - `6->12->6`, samples `16`, epochs `2`, max abs `2.07259142e-09`, final MSE `0.090901162`
    - `8->16->8`, samples `16`, epochs `1`, max abs `1.3349893e-09`, final MSE `0.124669507`
- Added first CartPole/Pendulum-shaped RTL trace sweep:
  - script: `scripts/exp_rtl_task_shape_trace_sweep.py`
  - summary: `runs/rtl_task_shape_trace_sweep_summary.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_task_shape_trace_sweep.py`
  - cases passed:
    - CartPole value/policy shape `4->16->2`: max abs `7.38049913e-10`, final MSE `0.071451252`
    - Pendulum actor shape `3->16->1`: max abs `1.20858831e-09`, final MSE `0.139430820`
    - Pendulum critic shape `4->16->1`: max abs `1.28952729e-09`, final MSE `0.094898385`
  - This is still a deterministic supervised trace, not a full environment-loop RL trace. Its purpose is to verify that the RTL/Python tick agreement holds on CartPole/Pendulum network geometry before using frozen RL batches.
- Added frozen CartPole/Pendulum environment-loop trace:
  - file-driven RTL testbench: `tb/tb_supervised_file_trace.sv`
  - checker: `scripts/check_supervised_file_trace.py`
  - exporter/runner: `scripts/exp_rtl_frozen_env_trace.py`
  - summary: `runs/rtl_frozen_env_trace_summary.csv`
  - frozen data:
    - `runs/rtl_frozen_env_trace/data/cartpole_value_frozen_seed42.dat`
    - `runs/rtl_frozen_env_trace/data/pendulum_actor_frozen_seed42.dat`
    - `runs/rtl_frozen_env_trace/data/pendulum_critic_frozen_seed42.dat`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_frozen_env_trace.py`
  - cases passed:
    - CartPole value frozen env samples, shape `4->16->2`: max abs `4.06842654e-09`, final MSE `0.249856610`
    - Pendulum actor frozen env samples, shape `3->16->1`: max abs `2.14315032e-09`, final MSE `0.298410560`
    - Pendulum critic frozen env samples, shape `4->16->1`: max abs `4.29959952e-09`, final MSE `0.016835963`
  - The rows come from actual Gym/Gymnasium environment transitions:
    - CartPole value rows: observation plus one-step action-value target for the sampled action.
    - Pendulum actor rows: observation plus normalized sampled continuous action.
    - Pendulum critic rows: observation/action plus scaled one-step reward target.
- Added true frozen runner update-batch trace:
  - exporter/runner: `scripts/exp_rtl_runner_batch_trace.py`
  - summary: `runs/rtl_runner_batch_trace_summary.csv`
  - frozen data:
    - `runs/rtl_runner_batch_trace/data/cartpole_value_td_batch_seed42.dat`
    - `runs/rtl_runner_batch_trace/data/cartpole_policy_aif_batch_seed42.dat`
    - `runs/rtl_runner_batch_trace/data/pendulum_pc_critic_td_batch_seed42.dat`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_runner_batch_trace.py`
  - cases passed:
    - CartPole value TD batch, shape `4->16->2`: max abs `8.20470414e-09`, final MSE `0.242037104`
    - CartPole policy AIF target-logit batch, shape `4->16->2`: max abs `5.7368033e-08`, final MSE `0.004131879`
    - Pendulum PC critic TD batch, shape `4->16->1`: max abs `3.13335337e-10`, final MSE `0.001333264`
  - These batches use the same target equations as the live runners:
    - CartPole value: `r + discount * sum(pi(next_state) * V_target(next_state))`, inserted into the sampled action slot.
    - CartPole policy: AIF-style target logits from value preferences using the runner's greedy-smoothed target convention.
    - Pendulum PC critic: DDPG target `reward + discount * (1-done) * critic_target(next_state, actor_target(next_state))`, normalized by `q_scale` to match `PCCritic.exactlocal_update`.
- Added first online/multi-update runner trace:
  - exporter/runner: `scripts/exp_rtl_runner_update_sequence.py`
  - summary: `runs/rtl_runner_update_sequence_summary.csv`
  - manifest: `runs/rtl_runner_update_sequence/data/cartpole_value_td_sequence_seed42_manifest.txt`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_runner_update_sequence.py`
  - sequence: four deterministic CartPole value TD update batches, each with 12 samples, using the runner's target equation.
  - cases passed:
    - update `0`: max abs `2.86839303e-08`, final MSE `0.258909244`
    - update `1`: max abs `2.87485574e-09`, final MSE `0.336477584`
    - update `2`: max abs `3.0218964e-09`, final MSE `0.349891823`
    - update `3`: max abs `2.79369883e-09`, final MSE `0.314877582`
  - Current implementation runs each update as a separate RTL sim invocation for isolation. A future persistent tracebench should carry one RTL network state across the full update sequence.
- Added persistent multi-update runner trace:
  - persistent RTL testbench: `tb/tb_supervised_sequence_file_trace.sv`
  - persistent checker: `scripts/check_supervised_sequence_file_trace.py`
  - exporter/runner: `scripts/exp_rtl_runner_persistent_sequence.py`
  - summary: `runs/rtl_runner_persistent_sequence_summary.csv`
  - output curves:
    - `runs/rtl_runner_persistent_sequence/cartpole_value_td_persistent_sequence_seed42.csv`
    - `runs/rtl_runner_persistent_sequence/cartpole_policy_aif_persistent_sequence_seed42.csv`
    - `runs/rtl_runner_persistent_sequence/pendulum_pc_critic_td_persistent_sequence_seed42.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_runner_persistent_sequence.py`
  - sequence: one RTL `pc_network_nlayer` instance carries state/weights across four deterministic update batches, each with 12 samples.
  - cases passed:
    - CartPole value TD persistent sequence, shape `4->16->2`: max abs `3.73348187e-08`, final MSE `0.243638748`
    - CartPole policy AIF persistent sequence, shape `4->16->2`: max abs `1.45879438e-07`, final MSE `0.001041235`
    - Pendulum PC critic TD persistent sequence, shape `4->16->1`: max abs `1.03066501e-09`, final MSE `0.001557986`
  - persistent MSE curves:
    - CartPole value: `0.525735244 -> 0.336966119 -> 0.252845903 -> 0.248685594 -> 0.243638748`
    - CartPole policy: `3.793230723 -> 0.534438511 -> 0.067734248 -> 0.009951652 -> 0.001041235`
    - Pendulum PC critic: `0.006267264 -> 0.001874826 -> 0.001810860 -> 0.000870987 -> 0.001557986`
  - Depth/width finding: `pc_network_nlayer` is structurally N-layer, but the current RTL `theta_init_pkg.sv` only provides nonzero random presets for `THETA_L0` and `THETA_L1`; layers `ul >= 2` instantiate with zero theta presets. Faithful 4-layer/two-hidden MuJoCo traces therefore need an initializer/preset upgrade before they can match the current two-hidden TD3/PC critics.
- Added deeper initializer/preset path:
  - generator: `scripts/gen_theta_init_pkg.py`
  - regenerated package: `rtl/includes/theta_init_pkg.sv`
  - new generation K_LUT: `[8, 16, 16, 32]`
  - RTL change: `rtl/nc_neuralcomputer.sv` now routes `THETA_L2` into `pc_network_nlayer` when `ul == 2 && NUM_LAYERS > 3`, while preserving zero theta presets for the top clamped layer / unsupported deeper layers.
  - Python checkers were updated to use matching expanded generation prefixes:
    - 3-layer traces use `gen_k_lut=[8,16,16]`
    - 4-layer traces use `gen_k_lut=[8,16,16,32]`
  - 3-layer persistent regression after this initializer change passed:
    - CartPole value TD persistent sequence: max abs `5.88510995e-09`, final MSE `0.246831646`
    - CartPole policy AIF persistent sequence: max abs `9.37988452e-08`, final MSE `0.002029657`
    - Pendulum PC critic TD persistent sequence: max abs `2.45820474e-09`, final MSE `0.001058727`
- Added first 4-layer/two-hidden persistent trace:
  - RTL testbench: `tb/tb_supervised_sequence_file_trace4.sv`
  - checker: `scripts/check_supervised_sequence_file_trace4.py`
  - runner: `scripts/exp_rtl_four_layer_sequence.py`
  - summary: `runs/rtl_four_layer_sequence_summary.csv`
  - current full-trace command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_four_layer_sequence.py --hidden-widths 16,32,64`
  - current 128 smoke command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_four_layer_sequence.py --hidden-widths 128 --data-kinds synthetic,halfcheetah_q,halfcheetah_aif --updates 1 --samples-per-update 1 --infer-ticks 1 --learn-ticks 1 --eval-ticks 1 --reuse-existing`
  - shapes: `1 -> H -> H -> 23`, matching HalfCheetah critic output width and state+action input width at reduced hidden sizes.
  - runner now accepts selectable widths/data kinds and can reuse existing CSV traces:
    - `--hidden-widths 16,32,64`
    - `--data-kinds synthetic,halfcheetah_q,halfcheetah_aif`
    - `--updates N`
    - `--samples-per-update N`
    - `--infer-ticks N`
    - `--learn-ticks N`
    - `--eval-ticks N`
    - `--reuse-existing`
    - `--verilator-build-jobs N`
    - `--verilator-verilate-jobs N`
    - `--verilator-output-split N`
    - `--verilator-output-split-cfuncs N`
    - `--no-cache-binaries`
  - `scripts/run_test.sh` now also honors these environment variables for large generated models:
    - `VERILATOR_MDIR`
    - `VERILATOR_REUSE_BINARY`
    - `VERILATOR_BUILD_JOBS`
    - `VERILATOR_VERILATE_JOBS`
    - `VERILATOR_THREADS`
    - `VERILATOR_OUTPUT_SPLIT`
    - `VERILATOR_OUTPUT_SPLIT_CFUNCS`
  - The four-layer runner now caches Verilated binaries per width and sequence shape in `obj_dir/trace4_*` directories by default. This avoids recompiling the same `H` network for synthetic/Q/AIF data and should cut a full three-case width sweep by roughly the number of data cases.
  - regenerated theta package was widened to generation K_LUT `[8, 256, 256, 32]`; Python checkers/oracles were updated to the same generation prefix. This supports hidden widths `128` and `256`.
  - cases passed:
    - width `16`, synthetic 4-layer regression: max abs `6.26897977e-07`, final MSE `0.002085381`
    - width `16`, actual HalfCheetah TD3-Q frozen batch: max abs `7.02192241e-05`, final MSE `0.728824927`
    - width `16`, actual HalfCheetah AIF `G=-Q` frozen batch: max abs `0.000182388266`, final MSE `0.261830507`
    - width `32`, synthetic 4-layer regression: max abs `6.5555467e-08`, final MSE `0.002214951`
    - width `32`, actual HalfCheetah TD3-Q frozen batch: max abs `0.000118721548`, final MSE `0.670309443`
    - width `32`, actual HalfCheetah AIF `G=-Q` frozen batch: max abs `0.000102077859`, final MSE `0.135951562`
    - width `64`, synthetic 4-layer regression: max abs `5.5355905e-07`, final MSE `0.003572798`
    - width `64`, actual HalfCheetah TD3-Q frozen batch: max abs `8.30568432e-05`, final MSE `0.618543396`
    - width `64`, actual HalfCheetah AIF `G=-Q` frozen batch: max abs `2.48188172e-05`, final MSE `0.054633508`
    - width `128` smoke, synthetic 4-layer regression, updates/samples/ticks `1/1/1`: max abs `1.24461404e-10`, final MSE `0.000004768`
    - width `128` smoke, actual HalfCheetah TD3-Q frozen batch, updates/samples/ticks `1/1/1`: max abs `7.25637955e-09`, final MSE `0.000052739`
    - width `128` smoke, actual HalfCheetah AIF `G=-Q` frozen batch, updates/samples/ticks `1/1/1`: max abs `1.5989312e-08`, final MSE `0.001386633`
    - width `128`, normal ticks `8/2/10`, `1 update x 1 sample`, synthetic: max abs `1.52381421e-07`, final MSE `0.000007240`
    - width `128`, normal ticks `8/2/10`, `1 update x 1 sample`, HalfCheetah TD3-Q: max abs `1.91700623e-05`, final MSE `0.018443976`
    - width `128`, normal ticks `8/2/10`, `1 update x 1 sample`, HalfCheetah AIF `G=-Q`: max abs `2.41815802e-05`, final MSE `0.142681403`
    - width `128`, normal ticks `8/2/10`, `1 update x 2 samples`, synthetic: max abs `1.24268052e-07`, final MSE `0.001793740`
    - width `128`, normal ticks `8/2/10`, `1 update x 2 samples`, HalfCheetah TD3-Q: max abs `0.000474436313`, final MSE `1.106590950`
    - width `128`, normal ticks `8/2/10`, `1 update x 2 samples`, HalfCheetah AIF `G=-Q`: max abs `6.66155995e-05`, final MSE `0.411250396`
    - width `128`, normal ticks `8/2/10`, `1 update x 3 samples`, synthetic: max abs `4.84808068e-07`, final MSE `0.002802972`
    - width `128`, normal ticks `8/2/10`, `1 update x 3 samples`, HalfCheetah TD3-Q: max abs `0.000214733524`, final MSE `0.534056672`
    - width `128`, normal ticks `8/2/10`, `1 update x 3 samples`, HalfCheetah AIF `G=-Q`: max abs `6.15506469e-05`, final MSE `0.351076445`
  - Raw HalfCheetah observations/actions required reducing the local trace learning rate from `alpha=0.02` to `alpha=0.002` for stable float32 RTL-vs-Python agreement. No observation normalization was used in the passing Q/AIF cases.
  - The AIF batch uses the runner's TD3/AIF target convention and clamps the RTL bottom state to the PC critic's internal normalized raw target, i.e. `target_raw = -target_G` under signed EFE semantics.
  - Width-scaling note:
    - Width `32` compiles and simulates, but Verilator compilation is already heavy: one `32` synthetic build took about `282s`, produced a `267.7 MB` generated model, and simulation took about `27s`.
    - Width `64` also passes, but the harness is now very heavy: one `64` AIF build took about `702s`, produced a `691.1 MB` generated model across `837` C++ files, and simulation took about `151s`.
    - Width `128` now instantiates and passes synthetic/Q/AIF at normal `8/2/10` ticks through `1 update x 3 samples` after widening theta to `[8,256,256,32]`. The full `2 updates x 6 samples x 8/2/10 ticks` simulation at width `128` compiled but was too slow for a quick probe.
    - Parallel Verilator build knobs were added after the width-64 pass. A hidden `256` minimal smoke was attempted with `1 update x 1 sample x 1/1/1 ticks`. The dataset was staged successfully, but Verilator did not finish codegen/build within a 15-minute diagnostic timeout and did not emit the build directory/binary. The visible diagnostic warning was only the wide-bus `WIDTHCONCAT` warning on clearing `x_obs_flat_all` (`32768` bits), not a PC dynamics mismatch.
    - The next clean probe is either `2 updates x 3 samples` at hidden `128`, or a slimmer/dedicated hidden `256` testbench that reduces TB-wide packed bus/codegen burden before trying the full `256` smoke again.
    - The next clean step is not just "try 256"; it is to make the RTL simulation harness more scalable or move to actual hardware/synthesis-style validation, then run full hidden `256`.
- After the full hidden-256 Verilator path proved too heavy, a compositional contract verification lane was added:
  - primitive RTL testbench: `tb/tb_neuron_contract_trace.sv`
  - checker: `scripts/check_neuron_contract_trace.py`
  - runner: `scripts/exp_rtl_contract_sweep.py`
  - summary: `runs/rtl_contract_sweep_summary.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_contract_sweep.py`
  - purpose: verify scaling-sensitive neuron contracts directly instead of requiring full-network Verilator elaboration for every large/deep shape.
  - cases passed:
    - `fanin_64`: `N=64`, `M=1`, linear, max abs `7.31845474e-09`
    - `fanin_128`: `N=128`, `M=1`, linear, max abs `4.90699772e-10`
    - `fanin_256`: `N=256`, `M=1`, linear, max abs `5.56695938e-10`
    - `backflow_64`: `N=1`, `M=64`, linear, max abs `4.65579991e-10`
    - `backflow_128`: `N=1`, `M=128`, linear, max abs `4.93107259e-10`
    - `backflow_256`: `N=1`, `M=256`, linear, max abs `4.65579991e-10`
    - `balanced_64`: `N=64`, `M=64`, linear, max abs `7.31845474e-09`
    - `relu_fanin_256`: `N=256`, `M=1`, ReLU, max abs `4.13123607e-10`
  - cache sizes were practical:
    - `obj_dir/contract_fanin_256`: about `161 MB`
    - `obj_dir/contract_backflow_256`: about `146 MB`
    - `obj_dir/contract_relu_fanin_256`: about `161 MB`
  - interpretation: the large fan-in and large backflow neuron contracts match the Python oracle at 256 scale. The full `1 -> 256 -> 256 -> 23` Verilator problem is therefore much more likely an all-at-once structural elaboration/build scaling issue than a local PC equation issue.
  - New scalable verification strategy:
    - use primitive contract tests for large fan-in/backflow
    - use small/deep `pc_network_nlayer` traces for scheduler/boundary timing
    - use reduced-width full network traces for end-to-end learning curves
    - reserve full 256 all-at-once RTL for synthesis/hardware validation or a much more simulator-friendly implementation.
- Added the next compositional tier: `pc_layer` tile contracts.
  - layer-tile RTL testbench: `tb/tb_layer_contract_trace.sv`
  - checker: `scripts/check_layer_contract_trace.py`
  - runner: `scripts/exp_rtl_layer_contract_sweep.py`
  - summary: `runs/rtl_layer_contract_sweep_summary.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_layer_contract_sweep.py`
  - purpose: verify several neurons running in parallel with shared 256-wide presynaptic input or 256-wide backflow, without elaborating the full 256/256 network.
  - cases passed:
    - `tile_k2_fanin_256`: `K=2`, `N=256`, `M=1`, linear, max abs `6.13927841e-11`
    - `tile_k4_fanin_256`: `K=4`, `N=256`, `M=1`, linear, max abs `6.13927841e-11`
    - `tile_k8_fanin_256`: `K=8`, `N=256`, `M=1`, linear, max abs `6.64964318e-11`
    - `tile_k2_backflow_256`: `K=2`, `N=1`, `M=256`, linear, max abs `5.43147326e-11`
    - `tile_k4_backflow_256`: `K=4`, `N=1`, `M=256`, linear, max abs `5.99771738e-11`
    - `tile_k4_balanced_64`: `K=4`, `N=64`, `M=64`, linear, max abs `1.32992864e-10`
    - `tile_k4_relu_fanin_256`: `K=4`, `N=256`, `M=1`, ReLU, max abs `0`
  - cache sizes remained practical:
    - `obj_dir/layer_contract_tile_k4_fanin_256`: about `219 MB`
    - `obj_dir/layer_contract_tile_k8_fanin_256`: about `293 MB`
    - `obj_dir/layer_contract_tile_k4_backflow_256`: about `184 MB`
  - interpretation: this bridges the gap between single-neuron contracts and whole-network composition. Multiple RTL neurons running in parallel over 256-wide inputs/backflow match the Python oracle. Full 256/256 Verilator remains unnecessary as the primary proof vehicle unless the simulation model is redesigned for whole-fabric compilation.
- Added narrow/deep scheduler contract traces:
  - RTL testbench: `tb/tb_deep_scheduler_trace.sv`
  - checker: `scripts/check_deep_scheduler_trace.py`
  - runner: `scripts/exp_rtl_deep_scheduler_sweep.py`
  - summary: `runs/rtl_deep_scheduler_sweep_summary.csv`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/exp_rtl_deep_scheduler_sweep.py --tol 1e-4`
  - shapes passed:
    - `1->8->8->8->23`, five layers, max abs `4.96509573e-05`
    - `1->16->16->16->23`, five layers, max abs `5.59074529e-05`
  - cache sizes:
    - `obj_dir/deep_scheduler_h8`: about `260 MB`
    - `obj_dir/deep_scheduler_h16`: about `364 MB`
  - interpretation: multi-layer simultaneous tick scheduling and boundary timing compose across deeper networks without requiring full 256 width.
- Added initializer/index mapping check:
  - script: `scripts/check_theta_init_mapping.py`
  - command: `/home/gregoryv/miniconda3/envs/dsl2/bin/python scripts/check_theta_init_mapping.py`
  - result: `PASS theta_init_pkg selected indices match Python RNG mapping`
  - probes: `15`, failures: `0`
  - checked selected boundary and mid indices for `THETA_L0`, `THETA_L1`, and `THETA_L2`, including index `0`, `127`, `255` where applicable.
  - interpretation: the generated theta package matches the Python RNG recipe bit-for-bit at selected full-256 boundary indices, reducing risk of hidden init/index mapping errors in the compositional argument.
- Added full-fabric 256/256 structural elaboration proof:
  - synthesis/elaboration top: `rtl/synth_pcnet_256_top.sv`
  - Yosys script/checker was attempted:
    - `scripts/synth_pcnet_256_yosys.ys`
    - `scripts/check_synth_pcnet_256.py`
    - installed Yosys is old (`0.9`) and cannot parse the repo's SystemVerilog style (`input logic` in `rtl/hf_mac32.sv`), so Yosys is not the usable proof vehicle in this environment without translation/tool upgrade.
  - Verilator structural elaboration, no C++ build, succeeded:
    - command class: `verilator --xml-only --stats ... --top-module synth_pcnet_256_top`
    - summary: `runs/synth/pcnet_256_elaboration_summary.md`
    - XML: `runs/synth/pcnet_256.xml`
    - log: `runs/synth/pcnet_256_verilator_xml.log`
    - stats: `obj_dir/verilator_xml_256/Vsynth_pcnet_256_top__stats.txt`
  - key result:
    - shape: `1 -> 256 -> 256 -> 23`
    - Verilog modules read: `50`
    - C++ files built: `0`
    - walltime: `121.089s`
    - elaboration: `23.112s`
    - conversion: `83.469s`
    - peak allocation: `9374.984 MB`
    - XML size: `844 MB`
    - unrolled iterations: `539880`
    - unrolled loops: `4307`
  - interpretation: the full 256/256 RTL fabric structurally elaborates without width/index/generate failure. The previous failures are specifically the full C++ simulation build path, not RTL hierarchy elaboration.

Interpretation:

- The single-neuron RTL primitive is now directly verified against the Python equation oracle.
- Frozen CartPole/Pendulum environment-loop samples, true frozen runner update batches, isolated multi-update traces, persistent carried-state sequences, and actual HalfCheetah TD3-Q/AIF frozen batches now pass RTL-vs-Python learning-curve alignment at hidden widths `16`, `32`, and `64`, plus a minimal hidden `128` smoke. The next step toward full HalfCheetah is lengthening the `128` trace, then hidden `256`, followed by longer/more update sequences.
- Scaling note: RTL already has `pc_network_nlayer`, and Python now has `PCNetNLayer`. Deterministic 3-layer network tracing, a small rate/activation sweep, deterministic fixed-dataset supervised learning alignment, schedule/rate sweeps, larger supervised config sweeps, CartPole/Pendulum-shaped supervised traces, frozen environment-sample traces, frozen runner-batch traces, isolated runner update sequences, persistent runner update sequences, and reduced-width 4-layer traces pass.
- Important conceptual boundary: the existing RTL implements the paper’s local predictive-coding tick dynamics. The MuJoCo HalfCheetah success currently uses the BP-equivalent PC bridge. To physically verify the full inference-engine story, either show the RTL-faithful tick dynamics can reproduce the smaller tasks and then scale, or add the BP-equivalent/nudged mode into RTL as an explicitly verified hardware mode.

## Constraints / Tooling Note

This local environment does **not** currently expose a direct GitHub repository-creation action.
So if you want to push this repo, the easiest path is:

1. create an empty GitHub repo in the browser
2. add it as `origin`
3. push the local repo

Everything needed for the code and the chat handoff is now inside this repository.
