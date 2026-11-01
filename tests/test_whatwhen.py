"""CPU tests for the WHAT×WHEN core. Run: python -m tests.test_whatwhen  (or pytest)."""
import math

import torch

from ww.whatwhen import (WhatWhenMixer, fejer_kernel, fejer_weights, harmonics_angle, kron,
                         ww_chunked, ww_sequential, ww_step)

torch.manual_seed(0)
D = torch.float64


def rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def random_inputs(B=2, H=3, T=45, V=1, decay=False, M0=False, R=2):
    """Realistic keys/queries: Kronecker products of unit Fejér-root features."""
    w = torch.tensor(fejer_weights(R), dtype=D)
    th_k, th_q = torch.rand(B, H, T, dtype=D) * 6.3, torch.rand(B, H, T, dtype=D) * 6.3
    clock = torch.cumsum(torch.rand(B, H, T, dtype=D), -1)
    k = kron(harmonics_angle(th_k, w, R), harmonics_angle(clock, w, R))
    q = kron(harmonics_angle(th_q, w, R), harmonics_angle(clock - 0.7, w, R))
    v = torch.randn(B, H, T, V, dtype=D)
    la = -torch.rand(B, H, T, dtype=D) * 0.3 if decay else None
    m0 = torch.randn(B, H, k.shape[-1], V, dtype=D) * 0.3 if M0 else None
    return k, q, v, la, m0


def test_fejer():
    R = 2
    w = torch.tensor(fejer_weights(R), dtype=D)
    a, b = torch.rand(100, dtype=D) * 10, torch.rand(100, dtype=D) * 10
    fa, fb = harmonics_angle(a, w, R), harmonics_angle(b, w, R)
    assert torch.allclose((fa * fb).sum(-1), fejer_kernel(a - b, R), atol=1e-12)
    assert torch.allclose(fa.norm(dim=-1), torch.ones(100, dtype=D), atol=1e-12)
    assert abs(fejer_kernel(torch.tensor(math.pi, dtype=D), R).item() - 1 / 9) < 1e-12


def test_chunked_equals_sequential():
    for V in (1, 4):
        for decay in (False, True):
            for M0 in (False, True):
                for C in (8, 16):
                    k, q, v, la, m0 = random_inputs(V=V, decay=decay, M0=M0)
                    y_s, M_s = ww_sequential(k, q, v, la, m0)
                    y_c, M_c = ww_chunked(k, q, v, la, m0, chunk=C)
                    assert rel(y_c, y_s) < 1e-10, (V, decay, M0, C, rel(y_c, y_s))
                    assert rel(M_c, M_s) < 1e-10, (V, decay, M0, C, rel(M_c, M_s))


def test_chunked_gradients_equal_sequential():
    for decay in (False, True):
        k, q, v, la, m0 = random_inputs(T=37, V=2, decay=decay, M0=True)
        leaves = [t.clone().requires_grad_(True) for t in (k, q, v, m0)]
        la_s = la.clone().requires_grad_(True) if decay else None
        la_c = la.clone().requires_grad_(True) if decay else None
        leaves2 = [t.clone().requires_grad_(True) for t in (k, q, v, m0)]
        g_out = torch.randn(2, 3, 37, 2, dtype=D)
        g_M = torch.randn(2, 3, 25, 2, dtype=D)
        y, M = ww_sequential(leaves[0], leaves[1], leaves[2], la_s, leaves[3])
        ((y * g_out).sum() + (M * g_M).sum()).backward()
        y2, M2 = ww_chunked(leaves2[0], leaves2[1], leaves2[2], la_c, leaves2[3], chunk=8)
        ((y2 * g_out).sum() + (M2 * g_M).sum()).backward()
        for a, b in zip(leaves, leaves2):
            assert rel(b.grad, a.grad) < 1e-9, rel(b.grad, a.grad)
        if decay:
            assert rel(la_c.grad, la_s.grad) < 1e-9


def test_step_equals_sequential():
    k, q, v, la, m0 = random_inputs(T=20, V=3, decay=True, M0=True)
    y_s, M_s = ww_sequential(k, q, v, la, m0)
    M = m0
    ys = []
    for t in range(20):
        y, M = ww_step(k[:, :, t], q[:, :, t], v[:, :, t], la[:, :, t], M)
        ys.append(y)
    assert rel(torch.stack(ys, 2), y_s) < 1e-12 and rel(M, M_s) < 1e-12


def test_beta1_writes_exactly():
    """After writing (k, v) the memory returns exactly v at address k (beta = 1, no decay)."""
    k, q, v, _, m0 = random_inputs(T=10, V=1, M0=True)
    _, M = ww_sequential(k, k, v, None, m0)
    readback = torch.einsum("bhk,bhkv->bhv", k[:, :, -1], M)
    assert torch.allclose(readback, v[:, :, -1], atol=1e-12)


def make_mixer(**kw):
    torch.manual_seed(1)
    m = WhatWhenMixer(d_model=16, n_heads=5, chunk=8, **kw).double()
    m.core_dtype = D
    torch.nn.init.normal_(m.out_proj.weight, std=0.1)  # non-zero so outputs are informative
    return m


def test_mixer_paths_agree():
    """chunked == sequential; split with state carry == one pass == token-by-token decode."""
    for kw in ({}, {"value_dim": 3}, {"decay": True}, {"when": "constant"}, {"when": "none"},
               {"what": "none"}, {"tick_param": "raw"}):
        m = make_mixer(**kw)
        x = torch.randn(2, 29, 16, dtype=D)
        y_c, st = m(x, want_state=True)
        y_s, _ = m(x, want_state=True, core="sequential")
        assert rel(y_c, y_s) < 1e-10, (kw, rel(y_c, y_s))
        y1, s1 = m(x[:, :13], want_state=True)
        y2, s2 = m(x[:, 13:], state=s1, want_state=True)
        assert rel(torch.cat([y1, y2], 1), y_c) < 1e-10, kw
        assert rel(s2["M"], st["M"]) < 1e-10
        s, outs = None, []
        for t in range(29):
            if s is None:
                yt, s = m(x[:, :1], want_state=True)
            else:
                yt, s = m(x[:, t:t + 1], state=s, want_state=True)
            outs.append(yt)
        assert rel(torch.cat(outs, 1), y_c) < 1e-10, (kw, rel(torch.cat(outs, 1), y_c))


def test_mixer_causal():
    m = make_mixer()
    x = torch.randn(1, 40, 16, dtype=D)
    y, _ = m(x)
    x2 = x.clone()
    x2[:, 25:] += torch.randn_like(x2[:, 25:])
    y2, _ = m(x2)
    assert torch.allclose(y[:, :25], y2[:, :25], atol=1e-12)
    assert not torch.allclose(y[:, 25:], y2[:, 25:])


def test_constant_clock_is_position():
    m = make_mixer(when="constant")
    x = torch.randn(1, 12, 16, dtype=D)
    ctrl = m.in_proj(x).view(1, 12, 5, m.n_ctrl)
    _, _, _, _, clock_T = m._features(ctrl, None)
    omega = torch.nn.functional.softplus(m.tick_bias)
    assert torch.allclose(clock_T[0], torch.remainder(12 * omega, 2 * math.pi), atol=1e-9)
    raw = make_mixer(when="constant", tick_param="raw")  # raw ticks start at the base frequencies
    _, _, _, _, clock_raw = raw._features(raw.in_proj(x).view(1, 12, 5, raw.n_ctrl), None)
    assert torch.allclose(clock_raw[0], torch.remainder(12 * raw.tick_bias, 2 * math.pi), atol=1e-9)


def test_long_clock_is_exact():
    """The fp64 clock stays exact over long contexts (no fp32 drift)."""
    m = make_mixer(when="constant")
    m.core_dtype = torch.float32
    m.float()
    x = torch.zeros(1, 20000, 16)
    ctrl = m.in_proj(x).view(1, 20000, 5, m.n_ctrl)
    _, _, _, _, clock_T = m._features(ctrl.float(), None)
    tick = torch.nn.functional.softplus(m.tick_bias.float()).double()  # the fp32 tick the model uses
    expect = torch.remainder(20000 * tick, 2 * math.pi)
    err = torch.remainder(clock_T[0] - expect + math.pi, 2 * math.pi) - math.pi
    assert err.abs().max() < 1e-4, err.abs().max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
