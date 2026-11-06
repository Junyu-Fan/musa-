# Optional MUSA fused RoPE

This adds one inference kernel on top of the existing FAST_ROTARY, BF16 Linear,
QK-LN and MUSA Graph paths. It combines half swapping and the multiply/add
operations for Q and K. It does not change model weights or normalization.
BF16 products are rounded separately before addition, as in the Torch path;
the build disables FMA contraction. The option is off by default.

Build on the MUSA server with the SDK matching your runtime:

```bash
export MUSA_HOME=/workspace/musa-5.2.0
export LD_LIBRARY_PATH=/workspace/mccl-stub:$MUSA_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/musa-
PY=/workspace/environments/musa-py312/bin/python
$PY -B -m esm.utils.musa_fused_rotary --output /workspace/build/libesm_rotary.so
export ESM_MUSA_ROTARY_LIBRARY=/workspace/build/libesm_rotary.so
$PY -B -m unittest discover -s /workspace/musa-/tests -p test_musa_fused_rotary.py -v
```

The build refuses to overwrite a library; choose a fresh output path for a new
build. Keep the compiled library for as long as any captured graph can use it.
No package installation, environment upgrade, or privileged container is needed.

Enable before importing ESM and warm up before capturing graphs:

```bash
export ESM_MUSA_FUSED_ROTARY=1
```

For SDK inference this can be combined with
`ESM3.from_pretrained(device="musa:0", musa_fast=True)` and the existing
`ESM_QK_LN_NATIVE_BF16=1` option. The kernel supports contiguous BF16/FP32 Q/K,
full-width rotary dimensions and non-interleaved rotation. Other layouts,
CPU/CUDA devices, and grad-enabled execution retain the original path.
An enabled eligible call with a missing/incompatible library raises an error.

Disable with `ESM_MUSA_FUSED_ROTARY=0` before starting a new process. Changing
the flag does not rewrite an already captured graph. Recreate the model/graph
cache when changing inference configuration.

Measure the same dtype, batch, input and output task with the flag off/on;
report cold-start and steady-state times separately. Validate all model output
heads and CPU reference cases, rather than inferring model quality from a
local RoPE microbenchmark.
