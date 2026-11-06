# MUSA residual and SwiGLU fusions

From the repository root, build with the MUSA SDK:

```sh
build_dir=$(mktemp -d)
for op in residual swiglu; do
    "$MUSA_HOME/bin/mcc" -O3 -ffp-contract=off -fPIC -shared \
        "esm/utils/csrc/musa_${op}.mu" -o "$build_dir/lib${op}.so" \
        -I"$MUSA_HOME/include" -L"$MUSA_HOME/lib" -lmusart || break
done
export ESM_MUSA_RESIDUAL_LIBRARY="$build_dir/libresidual.so"
export ESM_MUSA_SWIGLU_LIBRARY="$build_dir/libswiglu.so"
```

For SwiGLU, call `esm.utils.musa_fused_blocks.prepare_swiglu("musa:0")`
with the actual indexed device before warmup or graph capture. The table
uses the backend's BF16 SiLU results. Keep the library files available while
inference runs. Set the library variables and prepare the table before creating
any graphs. Recreate graphs when changing the fusion configuration.

Residual mode 0 preserves FP32 division followed by FP32 addition. Each
fusion falls back to its original implementation for unsupported inputs
or when its library is unset. See the separate RoPE guide for its switch.
