"""Optional inference-only MUSA RoPE kernel. Build explicitly before enabling it.

The library uses a small C ABI and needs neither PyTorch C++ headers nor Ninja.
Unsupported tensor layouts and grad-enabled calls keep the existing Torch path.
An explicitly enabled but missing/incompatible library raises a clear error.
"""
import argparse
import ctypes
import functools
import os
from pathlib import Path
import subprocess

import torch


@functools.lru_cache(maxsize=4)
def _load_library(path: str):
    if not path:
        raise RuntimeError(
            "ESM_MUSA_FUSED_ROTARY=1 requires ESM_MUSA_ROTARY_LIBRARY. "
            "Build with python -m esm.utils.musa_fused_rotary --output PATH."
        )
    library = ctypes.CDLL(str(Path(path).expanduser().resolve()))
    version = library.esm_rotary_abi_version
    version.argtypes = []
    version.restype = ctypes.c_int
    if version() != 1:
        raise RuntimeError("Incompatible ESM MUSA rotary library ABI; rebuild it.")
    launch = library.esm_rotary_pair
    launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    return library, launch  # Keep the library alive while graphs can reference it.


def fused_rotary(q, k, cos, sin):
    """Return fused Q/K outputs, or None when the original path is required."""
    if torch.is_grad_enabled() or q.device.type != "musa" or q.ndim != 4:
        return None
    if q.dtype not in (torch.bfloat16, torch.float32) or k.shape != q.shape:
        return None
    _, length, heads, dim = q.shape
    if dim % 2 or not 0 < q.numel() <= 2 ** 30:
        return None
    if cos.shape != (length, 1, dim) or sin.shape != cos.shape:
        return None
    if not all(x.device == q.device and x.dtype == q.dtype and x.is_contiguous()
               for x in (q, k, cos, sin)):
        return None
    _, launch = _load_library(os.environ.get("ESM_MUSA_ROTARY_LIBRARY", ""))
    with torch.musa.device(q.device):
        out_q, out_k = torch.empty_like(q), torch.empty_like(k)
        status = launch(
            q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            out_q.data_ptr(), out_k.data_ptr(), int(q.dtype == torch.float32),
            q.numel(), length, heads, dim,
            torch.musa.current_stream(q.device).musa_stream,
        )
    if status:
        raise RuntimeError(f"ESM MUSA rotary launch failed with status {status}")
    return out_q, out_k


def build(output: Path):
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    sdk = Path(os.environ.get("MUSA_HOME", "/usr/local/musa"))
    source = Path(__file__).with_name("csrc") / "musa_rotary.mu"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(sdk / "bin/mcc"), "-O3", "-ffp-contract=off", "-fPIC", "-shared",
        str(source), "-o", str(output), f"-I{sdk / 'include'}",
        f"-L{sdk / 'lib'}", "-lmusart",
    ], check=True)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
