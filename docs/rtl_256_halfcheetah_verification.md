# RTL-Faithful 256/256 HalfCheetah Verification Argument

## Claim

The current evidence supports a compositional RTL-faithful verification claim for the HalfCheetah-shaped predictive-coding fabric:

```text
1 -> 256 -> 256 -> 23
```

The claim is not that the full 256/256 fabric has been simulated end-to-end in Verilator. That path is impractical because the generated C++ model explodes. The claim is that the full fabric is verified by composition:

```text
local neuron contracts
+ layer tile contracts
+ deep scheduler contracts
+ theta/index mapping
+ reduced-width HalfCheetah RTL traces
+ full-fabric structural elaboration
```

Together these show that the 256/256 RTL shape instantiates and that its repeated local dynamics match the Python RTL-faithful model at the scaling-sensitive boundaries.

## Behavioral Anchor: Python 256/256 HalfCheetah

The full 256/256 HalfCheetah behavior is established in the Python/Torch MuJoCo runner.

Artifact:

```text
python_runs/halfcheetah_td3_pcpc_aggregate_100000.csv
```

Five-seed 100k-step aggregate:

```text
BP TD3 Q:
  mean final avg50: 3113.118
  mean final eval: 3796.312
  mean best eval: 4012.140

BP-AIF, G=-Q:
  mean final avg50: 3113.118
  mean final eval: 3796.312
  mean best eval: 4012.140

PC/PC AIF, BP-equivalent PC update:
  mean final avg50: 2856.203
  mean final eval: 3804.795
  mean best eval: 3804.795
```

Interpretation: the 256/256 PC/PC substrate reaches the strong HalfCheetah locomotion regime in the software experiment path.

Boundary: this result uses the BP-equivalent PC update path, while the RTL contracts below verify the local RTL-faithful PC tick dynamics.

## Reduced-Width End-To-End RTL HalfCheetah Traces

The four-layer RTL/Python trace harness verifies actual HalfCheetah-shaped frozen batches at reduced widths, including the same bottom/top geometry:

```text
1 -> H -> H -> 23
```

Core artifacts:

```text
tb/tb_supervised_sequence_file_trace4.sv
scripts/check_supervised_sequence_file_trace4.py
scripts/exp_rtl_four_layer_sequence.py
runs/rtl_four_layer_sequence_summary.csv
```

The most recent targeted run:

```text
case: halfcheetah_aif_td_batch_1_128_128_23
shape: 1 -> 128 -> 128 -> 23
updates: 2
samples per update: 3
max abs error: 6.15506469e-05
tol: 0.001
final MSE: 0.135908432
status: pass
```

Prior recorded runs in `CHAT_HANDOFF.md` also include width `16`, `32`, `64`, and `128` synthetic, TD3-Q, and AIF frozen HalfCheetah cases. These passed RTL-vs-Python learning-curve alignment, with width `128` reaching normal `8/2/10` ticks through `1 update x 3 samples` and AIF reaching `2 updates x 3 samples`.

Interpretation: the full learning-loop trace behavior is verified end-to-end at substantial reduced widths, including actual HalfCheetah target batches.

## 256-Scale Neuron Contracts

The primitive contract tests verify the scaling-sensitive local equations directly at 256 fan-in and 256 backflow.

Artifacts:

```text
tb/tb_neuron_contract_trace.sv
scripts/check_neuron_contract_trace.py
scripts/exp_rtl_contract_sweep.py
runs/rtl_contract_sweep_summary.csv
```

Passed cases:

```text
fanin_64        N=64,  M=1,   linear   max abs 7.31845474e-09
fanin_128       N=128, M=1,   linear   max abs 4.90699772e-10
fanin_256       N=256, M=1,   linear   max abs 5.56695938e-10
backflow_64     N=1,   M=64,  linear   max abs 4.65579991e-10
backflow_128    N=1,   M=128, linear   max abs 4.93107259e-10
backflow_256    N=1,   M=256, linear   max abs 4.65579991e-10
balanced_64     N=64,  M=64,  linear   max abs 7.31845474e-09
relu_fanin_256  N=256, M=1,   ReLU     max abs 4.13123607e-10
```

Interpretation: the RTL neuron implements the PC tick equations at the 256-wide boundaries needed by the HalfCheetah fabric:

```text
mu       = sum_j theta_j * phi(x_j) + bias
eps      = x_eff - mu
back     = theta * eps
theta   += alpha * eps * phi(x_j)
x       += gamma * (phi'(x_eff) * sum(back_in) - eps)
```

## 256-Scale Layer Tile Contracts

Layer tile contracts verify multiple RTL neurons running in parallel over shared 256-wide inputs or 256-wide backflow.

Artifacts:

```text
tb/tb_layer_contract_trace.sv
scripts/check_layer_contract_trace.py
scripts/exp_rtl_layer_contract_sweep.py
runs/rtl_layer_contract_sweep_summary.csv
```

Passed cases:

```text
tile_k2_fanin_256       K=2, N=256, M=1,   linear   max abs 6.13927841e-11
tile_k4_fanin_256       K=4, N=256, M=1,   linear   max abs 6.13927841e-11
tile_k8_fanin_256       K=8, N=256, M=1,   linear   max abs 6.64964318e-11
tile_k2_backflow_256    K=2, N=1,   M=256, linear   max abs 5.43147326e-11
tile_k4_backflow_256    K=4, N=1,   M=256, linear   max abs 5.99771738e-11
tile_k4_balanced_64     K=4, N=64,  M=64,  linear   max abs 1.32992864e-10
tile_k4_relu_fanin_256  K=4, N=256, M=1,   ReLU     max abs 0
```

Interpretation: this bridges single-neuron verification to layer-level composition. The repeated-neuron layer tile, back matrix generation, and transpose wiring match the Python oracle at 256-scale fan-in/backflow.

## Deep Scheduler Contracts

The scheduler contracts verify simultaneous tick boundary timing across deeper networks without width explosion.

Artifacts:

```text
tb/tb_deep_scheduler_trace.sv
scripts/check_deep_scheduler_trace.py
scripts/exp_rtl_deep_scheduler_sweep.py
runs/rtl_deep_scheduler_sweep_summary.csv
```

Passed cases:

```text
1 -> 8  -> 8  -> 8  -> 23   max abs 4.96509573e-05   tol 1e-4
1 -> 16 -> 16 -> 16 -> 23   max abs 5.59074529e-05   tol 1e-4
```

Interpretation: multi-layer `pc_network_nlayer` scheduling and boundary timing compose across depth. The observed errors are within the expected float32-vs-Python tolerance for deeper traces.

## Theta / Index Mapping

The theta initializer check verifies selected full-256 boundary indices in the generated RTL theta package against the Python RNG recipe.

Artifact:

```text
scripts/check_theta_init_mapping.py
```

Result:

```text
probes: 15
failures: 0
status: pass
```

Checked selected indices across:

```text
THETA_L0
THETA_L1
THETA_L2
```

including boundary and mid indices such as `0`, `127`, and `255` where applicable.

Interpretation: the full-width generated theta package matches the Python initialization mapping at selected boundary points, reducing risk of hidden off-by-one or generation-prefix errors.

## Full-Fabric Structural Elaboration

The full 256/256 RTL fabric structurally elaborates when using Verilator as an elaborator only, without building the C++ simulation model.

Artifacts:

```text
rtl/synth_pcnet_256_top.sv
runs/synth/pcnet_256_elaboration_summary.md
runs/synth/pcnet_256.xml
runs/synth/pcnet_256_verilator_xml.log
obj_dir/verilator_xml_256/Vsynth_pcnet_256_top__stats.txt
```

Shape:

```text
1 -> 256 -> 256 -> 23
```

Key result:

```text
PASS
Verilog modules read: 50
C++ files built: 0
Walltime: 121.089s
Peak allocation: 9374.984 MB
XML size: 844 MB
Unrolled iterations: 539880
Unrolled loops: 4307
```

Interpretation: the full RTL hierarchy elaborates without width, index, or generate failure. The previous crashes are specifically the full C++ simulation build path, not the RTL structure.

## Overall Status

The verification status is:

```text
Python 256/256 HalfCheetah behavior:        passed
RTL 128 HalfCheetah-shaped traces:          passed
RTL 256 neuron contracts:                   passed
RTL 256 layer-tile contracts:               passed
Deep scheduler contracts:                   passed
Theta/index mapping contracts:              passed
Full 256/256 structural elaboration:        passed
Full 256/256 Verilator simulation trace:    not attempted further / impractical
```

## Conclusion

The 256/256 RTL-faithful HalfCheetah story is now supported compositionally.

The strongest precise wording is:

> The full `1 -> 256 -> 256 -> 23` RTL fabric structurally elaborates, and its scaling-sensitive local PC dynamics are verified by 256-wide neuron contracts, 256-wide layer tile contracts, deep scheduler traces, theta/index mapping checks, and reduced-width HalfCheetah-shaped end-to-end RTL/Python traces. Full 256/256 HalfCheetah behavior is established in the Python PC/PC runner. A full all-neuron Verilator simulation remains impractical due to C++ model generation, but is no longer the central proof vehicle.

This is close to the complete compositional verification argument. A future direct full-fabric simulation would be an additional engineering artifact, not the core correctness proof.
