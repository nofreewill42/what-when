"""WHAT×WHEN memory: the recurrence, an exact chunkwise-parallel form, and the mixer layer.

One head is a delta-rule fast-weight memory with a K = (2R+1)^2 dimensional key space
(R = 2 harmonics -> K = 25) and a V-dimensional payload (V = 1 in the original model):

    key    k_t = f(theta_k[t]) (x) f(phi[t])               Kronecker product of two unit vectors
    query  q_t = f(theta_q[t]) (x) f(phi[t] + off[t])
    clock  phi[t] = sum_{s <= t} tick[s]                   per-head learned clock (WHEN)
    f(a)   = [w0, w1 cos a, w1 sin a, w2 cos 2a, w2 sin 2a]  Fejér-root harmonic features, |f| = 1

so k_i . q_j = Fejer(dtheta) * Fejer(dphi). The memory M (K x V) follows the beta = 1 delta rule,
optionally with a forget gate alpha_t in (0, 1] (an ablation; the original model has none):

    M_t = alpha_t M_{t-1} + k_t (v_t - alpha_t k_t^T M_{t-1})^T
    y_t = q_t^T M_t                                        (read after write)

Three implementations of the same recurrence live here:
  ww_sequential  token-by-token loop, the definition (used by the tests)
  ww_chunked     chunkwise-parallel (UT/WY form, as in DeltaNet), used for training and eval
  ww_step        one token, used for decoding

Chunkwise form (chunk of C tokens, starting state M0; gamma_i = prod_{j<=i} alpha_j inside the chunk):
    (I + A) u = v - gamma * (K M0),  A_ij = (gamma_i/gamma_j) k_i.k_j  (j < i)
    y        = gamma * (Q M0) + P u,  P_ij = (gamma_i/gamma_j) q_i.k_j  (j <= i)
    M_C      = gamma_C M0 + Kt^T u,   Kt_j = (gamma_C/gamma_j) k_j
With Tinv = (I + A)^-1, W = Tinv (gamma * K), U0 = Tinv v this becomes u = U0 - W M0 and
    M_C = (gamma_C I - Kt^T W) M0 + Kt^T U0
so the only sequential work is a K x K matrix times the K x V state per chunk.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------------------------
# Fejér-root harmonic features
# ----------------------------------------------------------------------------------------------
def fejer_weights(R: int = 2) -> list[float]:
    """Weights [w0, w1, w1, ..., wR, wR] that make f(a).f(b) the Fejér kernel F_R(a - b)."""
    norm = math.sqrt(R + 1)
    w = [1.0 / norm]
    for m in range(1, R + 1):
        c = math.sqrt(2.0 * (1.0 - m / (R + 1))) / norm
        w += [c, c]
    return w


def fejer_kernel(delta: torch.Tensor, R: int = 2) -> torch.Tensor:
    """F_R(delta) = 1/(R+1) * (1 + 2 sum_m (1 - m/(R+1)) cos(m delta)); equals 1 at 0."""
    out = torch.ones_like(delta)
    for m in range(1, R + 1):
        out = out + 2.0 * (1.0 - m / (R + 1)) * torch.cos(m * delta)
    return out / (R + 1)


def harmonics_cs(c: torch.Tensor, s: torch.Tensor, w: torch.Tensor, R: int) -> torch.Tensor:
    """Features from (cos a, sin a) via the angle-addition recurrence. Output [..., 2R+1].

    If (c, s) has norm r < 1 the feature norm is < 1 (never > 1), which keeps the delta rule stable.
    """
    feats = [torch.ones_like(c)]
    cm, sm = c, s
    for m in range(1, R + 1):
        if m > 1:
            cm, sm = cm * c - sm * s, sm * c + cm * s
        feats += [cm, sm]
    return torch.stack(feats, dim=-1) * w


def harmonics_angle(a: torch.Tensor, w: torch.Tensor, R: int) -> torch.Tensor:
    return harmonics_cs(torch.cos(a), torch.sin(a), w, R)


def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[..., F] x [..., F] -> [..., F*F], index = what_index * F + when_index."""
    return (a[..., :, None] * b[..., None, :]).flatten(-2)


# ----------------------------------------------------------------------------------------------
# The recurrence: reference, chunked, single step.  Layout: heads before time.
#   k, q: [B, H, T, K]   v: [B, H, T, V]   log_alpha: [B, H, T] or None   M0: [B, H, K, V] or None
# ----------------------------------------------------------------------------------------------
def ww_sequential(k, q, v, log_alpha=None, M0=None):
    B, H, T, K = k.shape
    V = v.shape[-1]
    M = k.new_zeros(B, H, K, V) if M0 is None else M0
    ys = []
    for t in range(T):
        if log_alpha is not None:
            M = M * torch.exp(log_alpha[:, :, t])[..., None, None]
        kt = k[:, :, t]
        old = torch.einsum("bhk,bhkv->bhv", kt, M)
        M = M + kt[..., :, None] * (v[:, :, t] - old)[..., None, :]
        ys.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], M))
    return torch.stack(ys, dim=2), M


def ww_step(k, q, v, log_alpha, M):
    """One token. k, q: [B, H, K]; v: [B, H, V]; log_alpha: [B, H] or None; M: [B, H, K, V]."""
    if log_alpha is not None:
        M = M * torch.exp(log_alpha)[..., None, None]
    old = torch.einsum("bhk,bhkv->bhv", k, M)
    M = M + k[..., :, None] * (v - old)[..., None, :]
    y = torch.einsum("bhk,bhkv->bhv", q, M)
    return y, M


def ww_chunked(k, q, v, log_alpha=None, M0=None, chunk: int = 32):
    B, H, T, K = k.shape
    V = v.shape[-1]
    C = chunk
    pad = (-T) % C
    if pad:  # zero keys/values write nothing; alpha = 1 keeps the state; outputs are cut off below
        k = F.pad(k, (0, 0, 0, pad))
        q = F.pad(q, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        if log_alpha is not None:
            log_alpha = F.pad(log_alpha, (0, pad))
    N = (T + pad) // C
    k = k.reshape(B, H, N, C, K)
    q = q.reshape(B, H, N, C, K)
    v = v.reshape(B, H, N, C, V)

    strict = torch.ones(C, C, dtype=torch.bool, device=k.device).tril(-1)
    incl = torch.ones(C, C, dtype=torch.bool, device=k.device).tril(0)
    kT = k.transpose(-1, -2)
    Gkk = k @ kT  # [B, H, N, C, C]
    Gqk = q @ kT
    if log_alpha is not None:
        G = log_alpha.reshape(B, H, N, C).cumsum(-1)  # log gamma_i
        decay = torch.exp((G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")))
        A = (Gkk * decay).masked_fill(~strict, 0.0)
        P = Gqk * decay  # already zero above the diagonal
        gam = torch.exp(G)[..., None]  # [B, H, N, C, 1]
        gamC = torch.exp(G[..., -1])[..., None, None]  # [B, H, N, 1, 1]
        Kt = k * torch.exp(G[..., -1:] - G)[..., None]  # (gamma_C / gamma_j) k_j
        Kg = k * gam
    else:
        A = Gkk.masked_fill(~strict, 0.0)
        P = Gqk.masked_fill(~incl, 0.0)
        gam, gamC, Kt, Kg = None, None, k, k

    eye_C = torch.eye(C, dtype=k.dtype, device=k.device)
    Tinv = torch.linalg.solve_triangular(A, eye_C, upper=False, unitriangular=True)  # (I + A)^-1
    # (unitriangular: A's zero diagonal is read as ones; the identity right-hand side broadcasts)
    W = Tinv @ Kg  # [B, H, N, C, K]
    U0 = Tinv @ v  # [B, H, N, C, V]
    KtT = Kt.transpose(-1, -2)
    eye_K = torch.eye(K, dtype=k.dtype, device=k.device)
    trans = (eye_K if gamC is None else gamC * eye_K) - KtT @ W  # [B, H, N, K, K]
    inject = KtT @ U0  # [B, H, N, K, V]

    M = k.new_zeros(B, H, K, V) if M0 is None else M0
    starts = []
    for n in range(N):  # the only sequential part: N steps of a K x K matmul
        starts.append(M)
        M = trans[:, :, n] @ M + inject[:, :, n]
    Ms = torch.stack(starts, dim=2)  # state at the start of each chunk [B, H, N, K, V]

    u = U0 - W @ Ms
    Qg = q if gam is None else q * gam
    y = Qg @ Ms + P @ u
    y = y.reshape(B, H, N * C, V)[:, :, :T]
    return y, M


# ----------------------------------------------------------------------------------------------
# The mixer layer (a drop-in replacement for attention in a pre-norm residual block)
# ----------------------------------------------------------------------------------------------
def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class WhatWhenMixer(nn.Module):
    """d_model -> per-head controls -> H WHAT×WHEN memories -> d_model.

    Controls per token per head (from one linear map of the residual stream):
        kx, ky   key WHAT direction (normalised to an angle)
        qx, qy   query WHAT direction
        tick     clock increment, tick = softplus(tick_bias_h + raw) > 0  (tick_param="softplus", default)
                 or tick = tick_bias_h + raw, unconstrained                (tick_param="raw")
        off      query clock offset, off = off_bias_h + raw (radians)
        v[V]     payload
        a        (decay ablation only) log alpha = -softplus(decay_bias_h + raw)

    Ablation switches:
        what = "learned" | "none"                 none: theta = 0 (time-only addressing)
        when = "learned" | "constant" | "none"    constant: tick = softplus(tick_bias_h), a learned
                                                  per-head constant (clock = position * omega_h);
                                                  none: phi = 0 (content-only addressing)
        decay = False | True
    The control layout and parameter count do not depend on what/when, so ablations stay matched.
    """

    def __init__(self, d_model: int, n_heads: int = 128, value_dim: int = 1, R: int = 2,
                 chunk: int = 32, what: str = "learned", when: str = "learned", decay: bool = False,
                 tick_param: str = "softplus", omega_min: float = TWO_PI / 2048, omega_max: float = 2.0):
        super().__init__()
        assert what in ("learned", "none")
        assert when in ("learned", "constant", "none")
        assert tick_param in ("softplus", "raw")
        self.tick_param = tick_param
        self.d_model, self.n_heads, self.value_dim, self.R = d_model, n_heads, value_dim, R
        self.F = 2 * R + 1
        self.K = self.F * self.F
        self.chunk, self.what, self.when, self.decay = chunk, what, when, decay
        self.n_ctrl = 6 + value_dim + (1 if decay else 0)
        self.in_proj = nn.Linear(d_model, n_heads * self.n_ctrl, bias=False)
        self.out_proj = nn.Linear(n_heads * value_dim, d_model, bias=False)
        self.out_proj.is_residual_out = True  # zero-initialised by the model
        omega = torch.exp(torch.linspace(math.log(omega_min), math.log(omega_max), n_heads))
        self.tick_bias = nn.Parameter(inverse_softplus(omega) if tick_param == "softplus" else omega)
        self.off_bias = nn.Parameter(-math.pi * torch.rand(n_heads))
        if decay:
            rate = torch.exp(torch.linspace(math.log(5e-4), math.log(0.1), n_heads))
            self.decay_bias = nn.Parameter(inverse_softplus(rate))
        self.register_buffer("fw", torch.tensor(fejer_weights(R)), persistent=False)
        self.core_dtype = torch.float32  # the memory always runs in fp32 (tests use fp64)

    def state_bytes(self) -> int:
        """Inference state per sequence: fp32 memory + fp64 clock."""
        return self.n_heads * (self.K * self.value_dim * 4 + 8)

    def _features(self, ctrl, clock0):
        """ctrl: [B, T, H, n_ctrl] fp32 -> k, q: [B, T, H, K]; v: [B, T, H, V]; log_alpha; clock_T."""
        B, T, H, _ = ctrl.shape
        w, R = self.fw.to(ctrl.dtype), self.R
        eps = 1e-6
        if self.what == "learned":
            kx, ky, qx, qy = ctrl[..., 0], ctrl[..., 1], ctrl[..., 2], ctrl[..., 3]
            rk = torch.sqrt(kx * kx + ky * ky + eps * eps)
            rq = torch.sqrt(qx * qx + qy * qy + eps * eps)
            fk_what = harmonics_cs(kx / rk, ky / rk, w, R)
            fq_what = harmonics_cs(qx / rq, qy / rq, w, R)
        else:
            one, zero = ctrl.new_ones(B, T, H), ctrl.new_zeros(B, T, H)
            fk_what = fq_what = harmonics_cs(one, zero, w, R)
        if self.when == "none":
            one, zero = ctrl.new_ones(B, T, H), ctrl.new_zeros(B, T, H)
            fk_when = fq_when = harmonics_cs(one, zero, w, R)
            clock_T = clock0
        else:
            pre = self.tick_bias.to(ctrl.dtype)
            pre = pre + ctrl[..., 4] if self.when == "learned" else pre.expand(B, T, H)
            tick = F.softplus(pre) if self.tick_param == "softplus" else pre
            # exact clock: cumulative sum in fp64, reduced mod 2*pi (gradient of remainder is 1)
            clock = torch.cumsum(tick.double(), dim=1)
            if clock0 is not None:
                clock = clock + clock0[:, None, :]
            clock = torch.remainder(clock, TWO_PI)
            clock_T = clock[:, -1]
            clock = clock.to(ctrl.dtype)
            off = self.off_bias.to(ctrl.dtype) + ctrl[..., 5]
            fk_when = harmonics_angle(clock, w, R)
            fq_when = harmonics_angle(clock + off, w, R)
        k = kron(fk_what, fk_when)
        q = kron(fq_what, fq_when)
        v = ctrl[..., 6:6 + self.value_dim]
        log_alpha = None
        if self.decay:
            log_alpha = -F.softplus(self.decay_bias.to(ctrl.dtype) + ctrl[..., 6 + self.value_dim])
        return k, q, v, log_alpha, clock_T

    def forward(self, x, state=None, want_state: bool = False, core: str = "chunked"):
        """x: [B, T, d]. state: None or {"M": [B,H,K,V] fp32, "clock": [B,H] fp64}."""
        B, T, _ = x.shape
        H, V = self.n_heads, self.value_dim
        ctrl = self.in_proj(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            ctrl = ctrl.to(self.core_dtype).view(B, T, H, self.n_ctrl)
            clock0 = None if state is None else state["clock"]
            M0 = None if state is None else state["M"]
            k, q, v, log_alpha, clock_T = self._features(ctrl, clock0)
            k, q, v = (t.permute(0, 2, 1, 3) for t in (k, q, v))  # [B, H, T, .]
            if log_alpha is not None:
                log_alpha = log_alpha.permute(0, 2, 1)
            if T == 1 and M0 is not None:
                la = None if log_alpha is None else log_alpha[:, :, 0]
                y, M = ww_step(k[:, :, 0], q[:, :, 0], v[:, :, 0], la, M0)
                y = y[:, :, None]
            elif core == "sequential":
                y, M = ww_sequential(k, q, v, log_alpha, M0)
            else:
                y, M = ww_chunked(k, q, v, log_alpha, M0, chunk=self.chunk)
            y = y.permute(0, 2, 1, 3).reshape(B, T, H * V)
        out = self.out_proj(y.to(self.out_proj.weight.dtype))  # autocast (if on) casts to bf16
        new_state = None
        if want_state:
            if clock_T is None:
                clock_T = torch.zeros(B, H, dtype=torch.float64, device=x.device)
            new_state = {"M": M, "clock": clock_T}
        return out, new_state
