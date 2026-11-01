"""Evaluation: validation BPC, and long-context BPC with per-position loss.

  python -m ww.evaluate longctx --run runs/<name> [--lengths 1024,4096,16384] [--max_chars 4194304]
  python -m ww.evaluate full    --run runs/<name>          # re-run the full validation BPC

Long-context protocol (the same text for every arm):
  * the same first `max_chars` characters of the validation stream are split into non-overlapping
    windows of length L; every window starts from an empty state / empty context;
  * models without full-attention layers (ww, gdn, deltanet, the hybrid whose attention layer was
    trained with a 256 window) run through the window in one pass;
  * models with full-attention layers (tpp) are scored with *strided* windows when L > T: the
    first T characters as usual, then every block of `stride` characters is predicted by a fresh
    forward pass over the T characters ending at that block, so each scored character sees between
    T - stride and T - 1 characters of context: exactly its training conditions, and the best a
    T-trained Transformer can do on a stream. (A single pass with a sliding attention window would
    be unfair to it: the window evicts the first tokens, which Transformers use as attention sinks.)
  * reported: BPC over all positions, BPC over positions >= T only (the part where long context can
    help), and a log-binned per-position loss curve.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from ww.data import CharData, load_paths, val_offsets, windows
from ww.model import LM

from ww.paths import RESULTS

LN2 = math.log(2.0)


def amp_context(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def eval_bpc(model, data: CharData, seq_len: int, offsets, batch_size: int, device) -> float:
    model.eval()
    nll, n = 0.0, 0
    for i in range(0, len(offsets), batch_size):
        x = windows(data, offsets[i:i + batch_size], seq_len).to(device)
        with amp_context(device):
            logits, _ = model(x[:, :-1])
        nll += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                               reduction="sum").item()
        n += x[:, 1:].numel()
    model.train()
    return nll / n / LN2


@torch.no_grad()
def per_position_nll(model, data: CharData, L: int, offsets, batch_size: int, device):
    """Mean NLL (nats) at each position 0..L-1 over the given windows."""
    model.eval()
    tot = torch.zeros(L, dtype=torch.float64)
    count = 0
    for i in range(0, len(offsets), batch_size):
        x = windows(data, offsets[i:i + batch_size], L).to(device)
        with amp_context(device):
            logits, _ = model(x[:, :-1])
        nll = F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none")  # [b, L]
        tot += nll.double().sum(0).cpu()
        count += x.shape[0]
    model.train()
    return (tot / max(count, 1)).numpy()


@torch.no_grad()
def per_position_nll_strided(model, data: CharData, L: int, offsets, T: int, stride: int, batch_size: int,
                             device):
    """Like per_position_nll, but every position is predicted from at most T characters of context
    (fresh forward passes over the T characters ending at each block of `stride` positions)."""
    model.eval()
    tot = torch.zeros(L, dtype=torch.float64)
    count = 0
    blocks = [(0, 0, min(T, L))]  # (input start a, first scored position s, end e): input = x[a:e]
    s = T
    while s < L:
        e = min(s + stride, L)
        blocks.append((e - T, s, e))
        s = e
    for i in range(0, len(offsets), batch_size):
        x = windows(data, offsets[i:i + batch_size], L).to(device)
        inp, tgt = x[:, :-1], x[:, 1:]
        nll = torch.empty(x.shape[0], L, dtype=torch.float64, device=device)
        for a, s, e in blocks:
            with amp_context(device):
                logits, _ = model(inp[:, a:e])
            part = F.cross_entropy(logits.float().transpose(1, 2), tgt[:, a:e], reduction="none")
            nll[:, s:e] = part[:, s - a:].double()
        tot += nll.sum(0).cpu()
        count += x.shape[0]
    model.train()
    return (tot / max(count, 1)).numpy()


def has_full_attention(model) -> bool:
    from ww.model import Attention
    return any(isinstance(m, Attention) and m.window is None for m in model.modules())


def log_bins(L: int, n_bins: int = 24):
    edges = np.unique(np.round(np.logspace(0, math.log10(L), n_bins + 1)).astype(int))
    edges[0] = 0
    edges[-1] = L
    return edges


def load_run(run_dir: str, device):
    ck = torch.load(os.path.join(run_dir, "final.pt"), map_location="cpu", weights_only=False)
    model = LM(ck["config"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def longctx(run_dir: str, lengths, max_chars: int, device, batch_chars: int = 65536, stride: int = 256):
    model, ck = load_run(run_dir, device)
    cfg, summary = ck["config"], ck["summary"]
    seq_len = cfg["train"]["seq_len"]
    val = CharData(load_paths()["val"])
    rows, curves = [], {}
    for L in lengths:
        offs = val_offsets(val.n, L, max_chars=max_chars)
        bs = max(1, batch_chars // L)
        if L > seq_len and has_full_attention(model):
            nll = per_position_nll_strided(model, val, L, offs, seq_len, min(stride, seq_len), bs, device)
        else:
            nll = per_position_nll(model, val, L, offs, bs, device)
        late = nll[seq_len:] if L > seq_len else np.array([])
        edges = log_bins(L)
        curves[str(L)] = {"edges": edges.tolist(),
                          "bpc": [float(nll[a:b].mean() / LN2) for a, b in zip(edges[:-1], edges[1:])]}
        rows.append({"run": summary["run"], "arm": summary["arm"], "seed": summary["seed"],
                     "stage": summary["stage"], "trained_at": summary.get("timestamp", ""), "L": L,
                     "n_windows": len(offs),
                     "bpc_all": float(nll.mean() / LN2),
                     "bpc_pos_ge_train_len": float(late.mean() / LN2) if late.size else float("nan"),
                     "bpc_last_1024": float(nll[-1024:].mean() / LN2)})
        print(f"{summary['run']}  L={L:6d}  bpc_all={rows[-1]['bpc_all']:.4f}  "
              f"bpc_pos>={seq_len}={rows[-1]['bpc_pos_ge_train_len']:.4f}")
    with open(os.path.join(run_dir, "longctx.json"), "w") as f:
        json.dump({"rows": rows, "curves": curves, "max_chars": max_chars}, f, indent=1)
    append_csv(os.path.join(RESULTS, "longctx.csv"), rows)
    return rows


def append_csv(path, rows):
    """Append rows to a results CSV. `trained_at` (the run's finish time) ties each row to one checkpoint,
    so a re-trained run is re-evaluated and its old rows are ignored (ww.queue A5, ww.gate). If the file
    has other columns, it is rewritten with the union of the columns."""
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        fields += [k for k in r if k not in fields]
    old = []
    if os.path.exists(path):
        with open(path) as f:
            rd = csv.DictReader(f)
            have = rd.fieldnames or []
            if have == fields:
                old = None
            else:
                old = list(rd)
                fields = have + [k for k in fields if k not in have]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if old is None:
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(old)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["longctx", "full"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--lengths", default="1024,4096,16384")
    ap.add_argument("--max_chars", type=int, default=4 * 1024 * 1024)
    ap.add_argument("--stride", type=int, default=256, help="strided evaluation for full attention")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    if a.what == "longctx":
        longctx(a.run, [int(x) for x in a.lengths.split(",")], a.max_chars, device, stride=a.stride)
    else:
        model, ck = load_run(a.run, device)
        val = CharData(load_paths()["val"])
        T = ck["config"]["train"]["seq_len"]
        print(eval_bpc(model, val, T, val_offsets(val.n, T), 32, device))


if __name__ == "__main__":
    main()
