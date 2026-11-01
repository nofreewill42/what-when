"""CPU tests for the backbone and the arms that run without fla. Run: python -m tests.test_model"""
import copy

import torch

from ww.arms import ARMS
from ww.model import LM, causal_attention, param_report

D = torch.float64


def tiny(arm, **over):
    """The arm's real config, shrunk so CPU tests are fast (same mixer kinds and switches)."""
    m = copy.deepcopy(ARMS[arm]["model"])
    m.update(d_model=32, ffn_hidden=48 if m["ffn"] != "none" else m["ffn_hidden"])
    m["attn"]["n_heads"] = 2
    m["swa"]["window"] = 8
    m["ww"]["n_heads"] = max(2, m["ww"]["n_heads"] // 32)
    m["ww"]["chunk"] = 8
    m.update(over)
    torch.manual_seed(0)
    model = LM(m).double()
    for mod in model.modules():  # memories in fp64 for exact comparisons
        if hasattr(mod, "core_dtype"):
            mod.core_dtype = D
        if getattr(mod, "is_residual_out", False):  # non-zero so every path matters
            torch.nn.init.normal_(mod.weight, std=0.05)
    return model


CPU_ARMS = [a for a, c in ARMS.items() if not c["gpu_only"]]


def test_forward_backward_and_causality():
    for arm in CPU_ARMS:
        model = tiny(arm)
        x = torch.randint(0, 128, (2, 40))
        logits, _ = model(x)
        assert logits.shape == (2, 40, 128)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), x.reshape(-1))
        loss.backward()
        unused = ("tick_bias", "off_bias") if model.cfg["ww"]["when"] == "none" else ()
        for n, p in model.named_parameters():
            if n.split(".")[-1] in unused:
                continue  # the clock is switched off in this ablation
            assert p.grad is not None and torch.isfinite(p.grad).all(), (arm, n)
        x2 = x.clone()
        x2[:, 25:] = torch.randint(0, 128, (2, 15))
        with torch.no_grad():
            a, _ = model(x)
            b, _ = model(x2)
        assert torch.allclose(a[:, :25], b[:, :25], atol=1e-10), f"{arm} is not causal"


def test_decode_matches_full_forward():
    """Prefill + one-token steps reproduce the full-sequence logits (KV cache, WW state, windows)."""
    for arm in ("tpp", "ww", "ww_decay", "ww_v8", "ww_hybrid", "ww_constclock"):
        model = tiny(arm)
        x = torch.randint(0, 128, (2, 30))
        with torch.no_grad():
            full, _ = model(x)
            logits, st = model(x[:, :11], want_state=True)
            outs = [logits]
            for t in range(11, 30):
                lt, st = model(x[:, t:t + 1], states=st, want_state=True)
                outs.append(lt)
        got = torch.cat(outs, 1)
        assert torch.allclose(got, full, atol=1e-9), (arm, (got - full).abs().max())


def test_sliding_window_eval_and_decode():
    model = tiny("tpp")
    x = torch.randint(0, 128, (1, 50))
    with torch.no_grad():
        model.set_eval_window(12)
        full, _ = model(x)
        logits, st = model(x[:, :20], want_state=True)
        outs = [logits]
        for t in range(20, 50):
            lt, st = model(x[:, t:t + 1], states=st, want_state=True)
            outs.append(lt)
        assert st[0]["k"].shape[2] == 12  # the cache never holds more than the window
        model.set_eval_window(None)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-9)
    with torch.no_grad():  # a window as long as the sequence changes nothing
        model.set_eval_window(50)
        a, _ = model(x)
        model.set_eval_window(None)
        b, _ = model(x)
    assert torch.allclose(a, b, atol=1e-12)


def test_blockwise_window_attention():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 4500, 8, dtype=D) for _ in range(3))
    got = causal_attention(q, k, v, window=300, block=1024)  # T > 4096 -> blockwise path
    i = torch.arange(4500)[:, None]
    j = torch.arange(4500)[None, :]
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i) & (i - j < 300))
    assert torch.allclose(got, ref, atol=1e-10)


def test_zero_init_residual_branches():
    torch.manual_seed(0)
    m = LM(ARMS["ww"]["model"])
    for mod in m.modules():
        if getattr(mod, "is_residual_out", False):
            assert mod.weight.abs().max() == 0
    rep = param_report(m)
    assert rep["params_embedding"] == 2 * 128 * 256



def test_strided_eval_gives_fresh_training_length_context():
    """Every position p >= T is scored by a pass over the T characters ending at its block."""
    import os
    import tempfile

    import numpy as np

    from ww.data import CharData, val_offsets
    from ww.evaluate import has_full_attention, per_position_nll, per_position_nll_strided
    model = tiny("tpp").float()
    assert has_full_attention(model) and not has_full_attention(tiny("ww_hybrid"))
    d = tempfile.mkdtemp()
    path = os.path.join(d, "v.u8")
    np.random.default_rng(0).integers(32, 127, 4000, dtype=np.uint8).tofile(path)
    data = CharData(path)
    T, L, stride = 16, 64, 1
    offs = val_offsets(data.n, L, max_windows=3)
    got = per_position_nll_strided(model, data, L, offs, T, stride, 3, torch.device("cpu"))
    x = torch.from_numpy(np.stack([np.asarray(data.data[o:o + L + 1], dtype=np.int64) for o in offs]))
    with torch.no_grad():
        for p in (0, 5, T - 1, T, 40, L - 1):
            a = max(0, p - T + 1)  # stride 1: the input window is the T characters ending at p
            logits, _ = model(x[:, a:p + 1])
            ref = torch.nn.functional.cross_entropy(logits[:, -1].float(), x[:, p + 1]).item()
            assert abs(got[p] - ref) < 1e-4, (p, got[p], ref)
    plain = per_position_nll(model, data, T, val_offsets(data.n, T, max_windows=3), 3, torch.device("cpu"))
    assert plain.shape == (T,)


def test_ww_flop_formula_matches_flop_counter():
    """The analytic WHAT×WHEN core FLOPs (used by gate Q4) equal torch's count of the executed matmuls."""
    from torch.utils.flop_counter import FlopCounterMode

    from ww.model import ww_core_flops_per_head
    from ww.whatwhen import ww_chunked
    for C, K, V in ((32, 25, 1), (16, 25, 8), (64, 9, 1)):
        T = 8 * C
        k = torch.randn(1, 1, T, K, dtype=torch.float64)
        k = k / k.norm(dim=-1, keepdim=True)
        q = torch.randn(1, 1, T, K, dtype=torch.float64)
        v = torch.randn(1, 1, T, V, dtype=torch.float64)
        with FlopCounterMode(display=False) as fc:
            ww_chunked(k, q, v, chunk=C)
        counted = fc.get_total_flops() / T
        analytic = ww_core_flops_per_head(C, K, V, solve=False)
        assert abs(counted / analytic - 1) < 1e-6, (C, K, V, counted, analytic)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
