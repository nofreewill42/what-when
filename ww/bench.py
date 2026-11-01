"""Throughput and decode benchmarks.

  python -m ww.bench train  --arms tpp,gdn,ww,...   # synthetic data: ms/update, targets/s, peak memory
                                                    # -> results/throughput.csv, results/micro_batch.json
  python -m ww.bench data                           # time to load one batch from the real training file
  python -m ww.bench decode --run runs/<name>       # batch 1 and 8 decode tok/s and memory
                                                    # -> results/decode.csv

The train benchmark tries micro-batches from 64 down, keeps the fastest one that fits in
--mem_limit_gb, and records it in results/micro_batch.json (train.py picks it up with
--micro_batch auto). Micro-batch size changes speed and memory only, never the update.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F

from ww.arms import load_arm
from ww.evaluate import amp_context, load_run
from ww.model import LM, decay_param_groups, param_report, state_bytes_per_seq, train_flops_per_token

from ww.paths import RESULTS


def _append_csv(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def time_updates(model, opt, T, B, mb, n_updates, device, vocab=128):
    accum = B // mb
    x_all = torch.randint(0, vocab, (B, T + 1), device=device)

    def one():
        for i in range(accum):
            x = x_all[i * mb:(i + 1) * mb]
            with amp_context(device):
                logits, _ = model(x[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, vocab), x[:, 1:].reshape(-1))
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(2):  # warmup (autotuning, allocator)
        one()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_updates):
        one()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / n_updates


def bench_train(arms, n_updates, mem_limit_gb, device):
    rows, chosen = [], {}
    mb_path = os.path.join(RESULTS, "micro_batch.json")
    if os.path.exists(mb_path):
        with open(mb_path) as f:
            chosen = json.load(f)
    for arm in arms:
        cfg = load_arm(arm)
        T, B = cfg["train"]["seq_len"], cfg["train"]["targets_per_update"] // cfg["train"]["seq_len"]
        best = None
        for mb in [m for m in (64, 32, 16, 8, 4, 2, 1) if B % m == 0]:
            torch.manual_seed(0)
            model = LM(cfg["model"]).to(device)
            opt = torch.optim.AdamW(decay_param_groups(model, 0.1), lr=1e-4, betas=(0.9, 0.95),
                                    fused=(device.type == "cuda"))
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            try:
                sec = time_updates(model, opt, T, B, mb, n_updates, device)
                peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            except torch.cuda.OutOfMemoryError:
                del model, opt
                torch.cuda.empty_cache()
                print(f"{arm}: micro_batch {mb} OOM")
                continue
            rep = param_report(model)
            row = {"arm": arm, "micro_batch": mb, "ms_per_update": round(sec * 1e3, 1),
                   "targets_per_sec": round(B * T / sec), "peak_mem_gb": round(peak, 2),
                   "fits": peak <= mem_limit_gb, "params_nonemb": rep["params_nonemb"],
                   "train_gflop_per_token": round(train_flops_per_token(model, T) / 1e9, 5),
                   "achieved_tflops": round(train_flops_per_token(model, T) * B * T / sec / 1e12, 2),
                   "state_kb_1k": round(state_bytes_per_seq(model, 1024) / 1024, 1),
                   "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
            rows.append(row)
            print(row, flush=True)
            del model, opt
            if row["fits"] and (best is None or row["ms_per_update"] < best["ms_per_update"]):
                best = row
            elif best is not None:
                break  # smaller micro-batches only get slower once a fitting one was slower
        if best is not None:
            chosen[arm] = best["micro_batch"]
    if rows:
        _append_csv(os.path.join(RESULTS, "throughput.csv"), rows)
    with open(mb_path, "w") as f:
        json.dump(chosen, f, indent=1)
    print("micro_batch choice:", chosen)


def bench_data(n_batches=20):
    import numpy as np

    from ww.data import CharData, load_paths, train_offsets, windows
    d = CharData(load_paths()["train"])
    t0 = time.perf_counter()
    for u in range(n_batches):
        windows(d, train_offsets(d.n, 1024, 64, 12345, u), 1024)
    sec = (time.perf_counter() - t0) / n_batches
    print(f"data: {sec * 1e3:.1f} ms per 64x1025 batch from {d.path} (cold or warm cache)")
    return sec


@torch.no_grad()
def bench_decode(run_dir, contexts, batches, new_tokens, device):
    model, ck = load_run(run_dir, device)
    model.eval()
    summary, seq_len = ck["summary"], ck["config"]["train"]["seq_len"]
    rows = []
    for ctx in contexts:
        for b in batches:
            model.set_eval_window(seq_len if ctx > seq_len else None)
            x = torch.randint(0, 128, (b, ctx), device=device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            try:
                with amp_context(device):
                    logits, states = model(x, want_state=True)
                    tok = logits[:, -1:].argmax(-1)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    for _ in range(new_tokens):
                        logits, states = model(tok, states=states, want_state=True)
                        tok = logits[:, -1:].argmax(-1)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                sec = time.perf_counter() - t0
            except NotImplementedError as e:
                print(f"{summary['arm']}: {e}")
                model.set_eval_window(None)
                return rows
            peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
            model.set_eval_window(None)
            full_kv = state_bytes_per_seq(model, ctx + new_tokens) / 1024
            model.set_eval_window(seq_len)
            windowed = state_bytes_per_seq(model, ctx + new_tokens) / 1024
            model.set_eval_window(None)
            rows.append({"run": summary["run"], "arm": summary["arm"], "context": ctx, "batch": b,
                         "tok_per_s": round(b * new_tokens / sec, 1),
                         "ms_per_step": round(sec / new_tokens * 1e3, 3), "peak_mem_mb": round(peak, 1),
                         "state_kb_per_seq_windowed": round(windowed, 1),
                         "state_kb_per_seq_fullkv": round(full_kv, 1), "eager": True})
            print(rows[-1], flush=True)
    if rows:
        _append_csv(os.path.join(RESULTS, "decode.csv"), rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "data", "decode"])
    ap.add_argument("--arms", default="tpp,gdn,ww")
    ap.add_argument("--updates", type=int, default=6)
    ap.add_argument("--mem_limit_gb", type=float, default=21.0)
    ap.add_argument("--run")
    ap.add_argument("--contexts", default="1024,4096,16384")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(RESULTS, exist_ok=True)
    if a.what == "train":
        bench_train(a.arms.split(","), a.updates, a.mem_limit_gb, device)
    elif a.what == "data":
        bench_data()
    else:
        bench_decode(a.run, [int(c) for c in a.contexts.split(",")],
                     [int(b) for b in a.batches.split(",")], a.new_tokens, device)


if __name__ == "__main__":
    main()
