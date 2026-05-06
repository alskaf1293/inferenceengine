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

The remaining gap is not that the whole project is impossible. It is that the first swapped PC value network is not yet matching the optimization strength of a backprop replay step inside the Millidge loop.

Likely contributors:

- insufficiently fixed-point PC inference during value updates
- sequential local PC replay updates are weaker than one batched Adam replay update
- the PC critic may need more optimizer-equivalent work per episode to match one backprop replay step

## Current Files To Use

If continuing this project, start from:

- `python_rtl/run_cartpole_millidge_bp.py`
- `python_rtl/run_cartpole_millidge_hybrid.py`

Use these modes in the hybrid runner:

- baseline reference: `--value-backend bp --policy-backend bp`
- first replacement: `--value-backend pc --policy-backend bp`

## Suggested Next Steps

1. Keep `bp/bp` as the reference check.
2. Continue tuning only `pc/bp` until it tracks the backprop baseline more closely.
3. Only swap the policy network after `pc/bp` is convincingly closer.
4. Focus tuning on:
   - stricter settling
   - more replay updates per episode
   - better PC value-target scaling / clipping
   - possible batch-style or teacher-style value distillation if needed

## Constraints / Tooling Note

This local environment does **not** currently expose a direct GitHub repository-creation action.
So if you want to push this repo, the easiest path is:

1. create an empty GitHub repo in the browser
2. add it as `origin`
3. push the local repo

Everything needed for the code and the chat handoff is now inside this repository.
