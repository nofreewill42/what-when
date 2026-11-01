"""One shared backbone for every arm; only the sequence mixer changes.

Block (pre-norm):  x = x + Mixer(RMSNorm(x));  x = x + FFN(RMSNorm(x))
LM:                embed -> blocks -> RMSNorm -> lm_head (untied)

Mixer kinds:
  attn      causal softmax attention, RoPE, optional QK-norm (the Transformer++ arm)
  swa       the same attention restricted to a sliding window (used once in the hybrid arm)
  ww        WHAT×WHEN memory (ww/whatwhen.py)
  gdn       Gated DeltaNet from flash-linear-attention (GPU only)
  deltanet  DeltaNet from flash-linear-attention (GPU only)

Init (all arms): normal(0, 0.02) for every Linear and the embedding, zeros for the output
projection of every residual branch (mixer out-proj, FFN down-proj), biases (if any) zero.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ww.whatwhen import WhatWhenMixer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


# ----------------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------------
def rope_cos_sin(pos0: int, T: int, head_dim: int, base: float, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim))
    t = torch.arange(pos0, pos0 + T, device=device, dtype=torch.float64)
    ang = torch.outer(t, inv)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x, cos, sin):  # x: [B, h, T, hd]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def causal_attention(q, k, v, window=None, block: int = 1024):
    """Causal attention; with a window, query i sees keys j with i - window < j <= i."""
    T = q.shape[2]
    if window is None or window >= T:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    dev = q.device
    if T <= 4096:
        i = torch.arange(T, device=dev)[:, None]
        j = torch.arange(T, device=dev)[None, :]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i) & (i - j < window))
    outs = []
    for s in range(0, T, block):  # blockwise so long evaluations never build a T x T mask
        e = min(T, s + block)
        ks = max(0, s - window + 1)
        i = torch.arange(s, e, device=dev)[:, None]
        j = torch.arange(ks, e, device=dev)[None, :]
        outs.append(F.scaled_dot_product_attention(
            q[:, :, s:e], k[:, :, ks:e], v[:, :, ks:e], attn_mask=(j <= i) & (i - j < window)))
    return torch.cat(outs, dim=2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_base: float = 10000.0, qk_norm: bool = True,
                 window: int | None = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.hd = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj.is_residual_out = True
        self.q_norm = RMSNorm(self.hd) if qk_norm else None
        self.k_norm = RMSNorm(self.hd) if qk_norm else None
        self.rope_base = rope_base
        self.window = window  # training-time window (None = full causal)
        self.eval_window = None  # evaluation override, e.g. sliding window = training length

    def active_window(self):
        return self.eval_window if self.eval_window is not None else self.window

    def kv_bytes_per_token(self) -> int:
        return 2 * self.h * self.hd * 2  # k and v, bf16

    def forward(self, x, state=None, want_state: bool = False):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        pos0 = 0 if state is None else state["pos"]
        cos, sin = rope_cos_sin(pos0, T, self.hd, self.rope_base, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        window = self.active_window()
        if state is None:
            y = causal_attention(q, k, v, window)
            k_keep, v_keep = k, v
        else:  # decoding one token against the cache
            assert T == 1, "cached attention supports one new token at a time"
            k_keep = torch.cat([state["k"], k], dim=2)
            v_keep = torch.cat([state["v"], v], dim=2)
            if window is not None:
                k_keep, v_keep = k_keep[:, :, -window:], v_keep[:, :, -window:]
            y = F.scaled_dot_product_attention(q, k_keep, v_keep)
        y = y.transpose(1, 2).reshape(B, T, self.h * self.hd)
        new_state = None
        if want_state:
            if window is not None:
                k_keep, v_keep = k_keep[:, :, -window:], v_keep[:, :, -window:]
            new_state = {"k": k_keep, "v": v_keep, "pos": pos0 + T}
        return self.o_proj(y), new_state


# ----------------------------------------------------------------------------------------------
# flash-linear-attention mixers (GPU only)
# ----------------------------------------------------------------------------------------------
class FLAMixer(nn.Module):
    def __init__(self, kind: str, d_model: int, n_heads: int = 4, head_dim: int = 64,
                 expand_v: float = 1.0, use_gate: bool = True, conv_size: int = 4, layer_idx: int = 0):
        super().__init__()
        import fla.layers as fl  # imported lazily so CPU tests work without fla
        self.kind, self.n_heads, self.head_dim, self.expand_v = kind, n_heads, head_dim, expand_v
        self.conv_size, self.d_model = conv_size, d_model
        if kind == "gdn":
            self.layer = fl.GatedDeltaNet(hidden_size=d_model, num_heads=n_heads, head_dim=head_dim,
                                          expand_v=expand_v, mode="chunk", use_gate=use_gate,
                                          use_short_conv=True, conv_size=conv_size, layer_idx=layer_idx)
        elif kind == "deltanet":
            assert head_dim * n_heads == d_model, "DeltaNet head_dim is d_model / n_heads"
            self.layer = fl.DeltaNet(hidden_size=d_model, num_heads=n_heads, expand_k=1.0,
                                     expand_v=expand_v, mode="chunk", use_beta=True, use_gate=False,
                                     use_short_conv=True, conv_size=conv_size, layer_idx=layer_idx)
        else:
            raise ValueError(kind)
        if hasattr(self.layer, "o_proj"):
            self.layer.o_proj.is_residual_out = True

    def state_bytes(self) -> int:
        dv = int(self.head_dim * self.expand_v)
        recurrent = self.n_heads * self.head_dim * dv * 4  # fp32 state
        conv = 3 * self.d_model * (self.conv_size - 1) * 2  # q/k/v short-conv caches, bf16
        return recurrent + conv

    def forward(self, x, state=None, want_state: bool = False):
        if state is not None or want_state:
            raise NotImplementedError("decoding is not wired for fla arms; use analytic state size")
        out = self.layer(x)
        return (out[0] if isinstance(out, tuple) else out), None


# ----------------------------------------------------------------------------------------------
# FFN, block, LM
# ----------------------------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, kind: str = "swiglu"):
        super().__init__()
        assert kind in ("swiglu", "relu", "relu2", "silu")
        self.kind = kind
        self.w_in = nn.Linear(d_model, (2 if kind == "swiglu" else 1) * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_model, bias=False)
        self.w_out.is_residual_out = True

    def forward(self, x):
        h = self.w_in(x)
        if self.kind == "swiglu":
            a, b = h.chunk(2, dim=-1)
            h = F.silu(a) * b
        elif self.kind == "relu":
            h = F.relu(h)
        elif self.kind == "relu2":
            h = F.relu(h).square()
        else:
            h = F.silu(h)
        return self.w_out(h)


def make_mixer(kind: str, cfg: dict, layer_idx: int) -> nn.Module:
    d = cfg["d_model"]
    if kind in ("attn", "swa"):
        a = cfg["attn"]
        window = cfg["swa"]["window"] if kind == "swa" else a.get("window")
        return Attention(d, a["n_heads"], a.get("rope_base", 10000.0), a.get("qk_norm", True), window)
    if kind == "ww":
        w = cfg["ww"]
        return WhatWhenMixer(d, n_heads=w["n_heads"], value_dim=w.get("value_dim", 1), R=w.get("R", 2),
                             chunk=w.get("chunk", 32), what=w.get("what", "learned"),
                             when=w.get("when", "learned"), decay=w.get("decay", False),
                             tick_param=w.get("tick_param", "softplus"))
    if kind in ("gdn", "deltanet"):
        f = cfg["fla"]
        return FLAMixer(kind, d, n_heads=f["n_heads"], head_dim=f["head_dim"],
                        expand_v=f.get("expand_v", 1.0), use_gate=f.get("use_gate", True),
                        conv_size=f.get("conv_size", 4), layer_idx=layer_idx)
    raise ValueError(f"unknown mixer {kind}")


class Block(nn.Module):
    def __init__(self, kind: str, cfg: dict, layer_idx: int):
        super().__init__()
        d = cfg["d_model"]
        self.kind = kind
        self.norm1 = RMSNorm(d)
        self.mixer = make_mixer(kind, cfg, layer_idx)
        self.ffn = None if cfg["ffn"] == "none" else FFN(d, cfg["ffn_hidden"], cfg["ffn"])
        self.norm2 = RMSNorm(d) if self.ffn is not None else None

    def forward(self, x, state=None, want_state: bool = False):
        h, st = self.mixer(self.norm1(x), state=state, want_state=want_state)
        x = x + h
        if self.ffn is not None:
            x = x + self.ffn(self.norm2(x))
        return x, st


class LM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d, vocab = cfg["d_model"], cfg.get("vocab", 128)
        layers = cfg["layers"]
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(kind, cfg, i) for i, kind in enumerate(layers)])
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.reset_parameters()
        if cfg.get("tie_embeddings", False):
            self.lm_head.weight = self.embed.weight

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "is_residual_out", False):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, states=None, want_state: bool = False):
        x = self.embed(idx)
        new_states = []
        for i, blk in enumerate(self.blocks):
            x, st = blk(x, None if states is None else states[i], want_state)
            new_states.append(st)
        logits = self.lm_head(self.norm_f(x))
        return logits, (new_states if want_state else None)

    def set_eval_window(self, window: int | None):
        """Full-attention layers attend to at most `window` past tokens (None restores training)."""
        for m in self.modules():
            if isinstance(m, Attention) and m.window is None:
                m.eval_window = window

    def mixer_kinds(self):
        return [b.kind for b in self.blocks]


# ----------------------------------------------------------------------------------------------
# Accounting
# ----------------------------------------------------------------------------------------------
def param_report(model: LM) -> dict:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    emb = model.embed.weight.numel()
    if model.lm_head.weight is not model.embed.weight:
        emb += model.lm_head.weight.numel()
    mixer = sum(p.numel() for b in model.blocks for p in b.mixer.parameters())
    ffn = sum(p.numel() for b in model.blocks if b.ffn is not None for p in b.ffn.parameters())
    return {"params_total": total, "params_embedding": emb, "params_nonemb": total - emb,
            "params_mixer": mixer, "params_ffn": ffn}


def state_bytes_per_seq(model: LM, context: int) -> int:
    """Inference memory per sequence at a given context: KV cache (bf16) or recurrent state."""
    total = 0
    for b in model.blocks:
        m = b.mixer
        if isinstance(m, Attention):
            w = m.active_window()
            total += m.kv_bytes_per_token() * (context if w is None else min(context, w))
        else:
            total += m.state_bytes()
    return total


def decay_param_groups(model: LM, weight_decay: float):
    """Weight decay on matrices (ndim >= 2) except the embedding; none on norms, biases, 1-D params."""
    decay, no_decay, seen = [], [], set()
    emb = model.embed.weight
    for p in model.parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (decay if (p.ndim >= 2 and p is not emb) else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def ww_core_flops_per_head(C: int, K: int, V: int, solve: bool = True) -> float:
    """Forward FLOPs per token per head of ww_chunked (matches torch's FlopCounterMode on the matmuls,
    tests/test_model.py). Per token: Gram matrices KK^T, QK^T 4CK; W = Tinv Kg, U0 = Tinv v 2C(K+V);
    trans = Kt^T W, inject = Kt^T U0 2K(K+V); state hand-off trans @ M 2K^2 V / C; outputs
    u = U0 - W M, Q M 4KV, P u 2CV; plus the unit-triangular inverse C^2/3 (intrinsic cost, C^3/3 per
    chunk; torch's generic solve does more and is not counted by FlopCounterMode)."""
    f = 4 * C * K + 2 * C * (K + V) + 2 * K * (K + V) + 2 * K * K * V / C + 4 * K * V + 2 * C * V
    return f + (C * C / 3 if solve else 0.0)


def train_flops_per_token(model: LM, seq_len: int) -> float:
    """Rough analytic training FLOPs per token (forward x 3). Matmul params: 6 per param.
    attention adds 6 * T * d per layer (causal average); WHAT×WHEN adds its chunked-core cost;
    fla layers are counted by their projections only (their core is O(d_head) per token per head).
    """
    n_matmul = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    flops = 6.0 * n_matmul
    for b in model.blocks:
        m = b.mixer
        if isinstance(m, Attention):
            w = m.active_window()
            ctx_avg = seq_len / 2 if (w is None or w >= seq_len) else w  # keys seen per query
            flops += 3.0 * 4.0 * ctx_avg * m.h * m.hd  # QK^T and AV, forward x 3
        elif isinstance(m, WhatWhenMixer):
            flops += 3.0 * ww_core_flops_per_head(m.chunk, m.K, m.value_dim) * m.n_heads
        elif isinstance(m, FLAMixer):
            dv = m.head_dim * m.expand_v
            flops += 3.0 * m.n_heads * (4 * 64 * (m.head_dim + dv) + 4 * m.head_dim * dv)
    return flops
