"""Run with the compiled library configured; GPU tests require one MUSA device."""
import os
import unittest
from unittest.mock import patch

import torch
try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

from esm.layers import rotary
from esm.utils.musa_fused_rotary import fused_rotary


def reference(q, k, cos, sin):
    def rotate(x):
        other = x.unflatten(-1, (2, x.shape[-1] // 2)).flip(-2).flatten(-2)
        return x * cos + other * sin
    return rotate(q), rotate(k)


GPU_READY = (torch_musa is not None and torch.musa.is_available()
             and bool(os.environ.get("ESM_MUSA_ROTARY_LIBRARY")))


class CPUFallback(unittest.TestCase):
    def test_cpu_does_not_load_native_library(self):
        x = torch.ones(1, 3, 2, 4)
        table = torch.ones(3, 1, 4)
        with torch.inference_mode(), patch.dict(os.environ, ESM_MUSA_ROTARY_LIBRARY=""):
            self.assertIsNone(fused_rotary(x, x, table, table))


@unittest.skipUnless(GPU_READY, "requires MUSA and ESM_MUSA_ROTARY_LIBRARY")
class FusedRotary(unittest.TestCase):
    def setUp(self):
        self.kick = torch.zeros(1, device="musa")

    def wait(self):
        _ = self.kick.cpu()

    def assert_bits(self, expected, actual):
        for a, b in zip(expected, actual):
            bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
            self.assertTrue(torch.equal(a.view(bits), b.view(bits)))

    def test_dtypes_batches_and_offset_tables(self):
        with torch.inference_mode():
            for dtype in (torch.bfloat16, torch.float32):
                for shape in ((1, 1, 24, 64), (2, 17, 3, 10), (1, 512, 24, 64)):
                    with self.subTest(dtype=dtype, shape=shape):
                        q = torch.randn(shape, device="musa", dtype=dtype)
                        k = torch.randn_like(q)
                        cos = torch.randn(shape[1] + 7, 1, shape[3], device="musa", dtype=dtype)[7:]
                        sin = torch.randn_like(cos)
                        self.assert_bits(reference(q, k, cos, sin), fused_rotary(q, k, cos, sin))

    def test_noncontiguous_and_grad_enabled_fallbacks(self):
        q = torch.randn(2, 11, 3, 64, device="musa").transpose(1, 2)
        table = torch.ones(3, 1, 64, device="musa")
        with torch.inference_mode():
            self.assertIsNone(fused_rotary(q, q, table, table))
        q = q.contiguous().requires_grad_()
        with torch.enable_grad():
            self.assertIsNone(fused_rotary(q, q, table, table))

    def test_module_preserves_offsets_and_interleaved_fallback(self):
        with torch.inference_mode():
            for interleaved in (False, True):
                layer = rotary.RotaryEmbedding(64, interleaved=interleaved).to("musa")
                q = torch.randn(2, 17, 3, 64, device="musa", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                with patch.object(rotary, "FUSED_ROTARY", False):
                    expected = layer(q, k, seqlen_offset=7)
                with patch.object(rotary, "FUSED_ROTARY", True):
                    actual = layer(q, k, seqlen_offset=7)
                self.assert_bits(expected, actual)

    def test_graph_replay_reads_updated_inputs(self):
        with torch.inference_mode():
            q = torch.randn(1, 17, 3, 64, device="musa", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            cos = torch.randn(17, 1, 64, device="musa", dtype=q.dtype)
            sin = torch.randn_like(cos)
            side = torch.musa.Stream()
            side.wait_stream(torch.musa.current_stream())
            with torch.musa.stream(side):
                for _ in range(3):
                    outputs = fused_rotary(q, k, cos, sin)
            torch.musa.current_stream().wait_stream(side)
            self.wait()
            graph = torch.musa.MUSAGraph()
            with torch.musa.stream(side):
                graph.capture_begin()
                try:
                    outputs = fused_rotary(q, k, cos, sin)
                finally:
                    graph.capture_end()
            torch.musa.current_stream().wait_stream(side)
            q.add_(0.5)
            k.sub_(0.25)
            graph.replay()
            self.wait()
            self.assert_bits(reference(q, k, cos, sin), outputs)


if __name__ == "__main__":
    unittest.main()
