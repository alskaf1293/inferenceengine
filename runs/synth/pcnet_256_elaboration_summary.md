# Full-Fabric 256/256 Elaboration Summary

Command class: Verilator structural elaboration only, no C++ simulation build.

Shape:

```text
1 -> 256 -> 256 -> 23
```

Top:

```text
rtl/synth_pcnet_256_top.sv
```

Result:

```text
PASS
```

Artifacts:

```text
runs/synth/pcnet_256.xml
runs/synth/pcnet_256_verilator_xml.log
obj_dir/verilator_xml_256/Vsynth_pcnet_256_top__stats.txt
```

Key Verilator report lines:

```text
Verilator 5.038 2025-07-08 rev v5.038
Built from 8.574 MB sources in 50 modules, into 0.000 MB in 0 C++ files needing 0.000 MB
Walltime 121.089 s (elab=23.112, cvt=83.469, bld=0.000)
alloced 9374.984 MB
```

Stats:

```text
Input, Verilog modules read: 50
Optimizations, Unrolled Iterations: 539880
Optimizations, Unrolled Loops: 4307
Peak stage memory: 9374.984 MB
XML size: 844 MB
```

Interpretation:

The full `1 -> 256 -> 256 -> 23` RTL fabric structurally elaborates without width/index/generate failure. This does not simulate the full fabric and does not build the C++ model that previously caused practical crashes; it proves the full hardware shape can be elaborated as RTL hierarchy.
