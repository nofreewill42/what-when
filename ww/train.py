"""Train one run and append it to results/ledger.csv.

  python -m ww.train --arm tpp --lr 2e-3 --seed 0 --updates 8192 --stage A2_r10
  python -m ww.train --arm ww  --lr auto --seed 0 --seconds 600  --stage A6_isotime

Budgets: --updates U (iso-token: every arm sees exactly U * 65,536 targets) or --seconds S (iso-time:
active training time, evaluation excluded). The schedule is a function of progress = u/U or t/S, so
warmup and cooldown always fit the actual run. Runs are idempotent: an existing finished run is
skipped unless --force; an interrupted iso-token run continues with --resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from ww.arms import ARMS, load_arm
from ww.data import CharData, load_paths, train_offsets, val_offsets, windows
from ww.evaluate import LN2, amp_context, eval_bpc
from ww.model import LM, decay_param_groups, param_report, state_bytes_per_seq, train_flops_per_token

from ww.paths import RESULTS, ROOT, RUNS

LEDGER = os.path.join(RESULTS, "ledger.csv")
LEDGER_FIELDS = [
    "run", "stage", "arm", "seed", "lr", "budget_kind", "budget", "updates", "targets",
    "active_seconds", "wall_seconds", "targets_per_sec", "data_wait_seconds", "peak_mem_gb",
    "micro_batch", "params_nonemb", "params_total", "state_kb_1k", "state_kb_16k",
    "train_gflop_per_token", "final_val_bpc", "final_quick_bpc", "status", "config_sha",
    "git_commit", "device", "torch", "timestamp",
]


def lr_factor(progress: float, schedule: str, warmup: float, cooldown: float) -> float:
    if progress < warmup:
        return (progress + 1e-9) / warmup
    if schedule == "cosine":
        p = min(1.0, (progress - warmup) / max(1e-9, 1.0 - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if schedule == "wsd":
        if progress < 1.0 - cooldown:
            return 1.0
        p = min(1.0, (progress - (1.0 - cooldown)) / cooldown)
        return max(0.0, 1.0 - math.sqrt(p))
    if schedule == "constant":
        return 1.0
    raise ValueError(schedule)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def resolve_lr(arm: str, cfg: dict, lr_arg):
    if lr_arg not in (None, "auto"):
        return float(lr_arg)
    if cfg["train"].get("lr") is not None:  # arms with a fixed recipe LR (tpp_legacy)
        return float(cfg["train"]["lr"])
    path = os.path.join(RESULTS, "lr_choice.json")
    if not os.path.exists(path):
        raise SystemExit("no results/lr_choice.json yet: run stage A1 and `python -m ww.gate select-lr`")
    with open(path) as f:
        choice = json.load(f)
    key = arm if arm in choice else (cfg.get("lr_from") or arm)
    if key not in choice:
        raise SystemExit(f"no tuned LR for {arm} (looked for {key!r}) in results/lr_choice.json")
    return float(choice[key])


def resolve_micro_batch(arm: str, cfg: dict, mb_arg):
    if mb_arg not in (None, "auto"):
        return int(mb_arg)
    path = os.path.join(RESULTS, "micro_batch.json")
    if os.path.exists(path):
        with open(path) as f:
            mb = json.load(f)
        if arm in mb:
            return int(mb[arm])
    return int(cfg["train"]["micro_batch"])


class Prefetcher:
    """Loads the next update's batch on a background thread (hides disk latency)."""

    def __init__(self, data: CharData, seq_len: int, batch: int, data_seed: int, start_update: int):
        self.q = queue.Queue(maxsize=3)
        self.stop = False

        def work():
            u = start_update
            while not self.stop:
                offs = train_offsets(data.n, seq_len, batch, data_seed, u)
                xb = windows(data, offs, seq_len)
                if torch.cuda.is_available():
                    xb = xb.pin_memory()
                while not self.stop:
                    try:
                        self.q.put((u, xb), timeout=0.5)
                        break
                    except queue.Full:
                        continue
                u += 1

        self.t = threading.Thread(target=work, daemon=True)
        self.t.start()

    def get(self, expected_update: int):
        u, xb = self.q.get()
        assert u == expected_update, (u, expected_update)
        return xb

    def close(self):
        self.stop = True


def append_ledger(row: dict):
    os.makedirs(RESULTS, exist_ok=True)
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


def run(arm: str, cfg: dict, lr: float, seed: int, stage: str, updates: int | None = None,
        seconds: float | None = None, micro_batch: int | None = None, device: str | None = None,
        paths: dict | None = None, runs_dir: str | None = None, force: bool = False,
        resume: bool = False, full_val: bool = True, quick_windows: int = 512, n_evals: int = 16,
        ckpt_minutes: float = 20.0, data_in_ram: bool = False, write_ledger: bool = True,
        val_max_windows: int | None = None) -> dict:
    assert (updates is None) != (seconds is None), "give exactly one of updates / seconds"
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tcfg, mcfg = cfg["train"], cfg["model"]
    T = tcfg["seq_len"]
    B = tcfg["targets_per_update"] // T
    mb = micro_batch or tcfg["micro_batch"]
    assert B % mb == 0, f"micro_batch {mb} must divide batch {B}"
    accum = B // mb
    budget_kind, budget = ("updates", updates) if updates is not None else ("seconds", seconds)
    tag = f"u{updates}" if updates is not None else f"t{int(seconds)}s"
    name = f"{stage}-{arm}-s{seed}-{tag}-lr{lr:.1e}"
    run_dir = os.path.join(runs_dir or RUNS, name)
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path) and not force:
        print(f"skip {name}: already finished")
        with open(summary_path) as f:
            return json.load(f)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LM(mcfg).to(device)
    preport = param_report(model)
    opt = torch.optim.AdamW(decay_param_groups(model, tcfg["weight_decay"]), lr=lr,
                            betas=tuple(tcfg["betas"]), eps=tcfg["eps"],
                            fused=(device.type == "cuda"))
    paths = paths or load_paths()
    train_data = CharData(paths["train"], mcfg["vocab"])
    val_data = CharData(paths["val"], mcfg["vocab"])
    if data_in_ram:
        train_data.data = np.fromfile(train_data.path, dtype=np.uint8)
    quick_offs = val_offsets(val_data.n, T, max_windows=quick_windows)
    full_offs = val_offsets(val_data.n, T, max_windows=val_max_windows)
    eval_bs = min(64, 2 * mb)  # no-grad evaluation fits at least twice the training micro-batch

    u, active, targets, data_wait = 0, 0.0, 0, 0.0
    history = []
    last_ckpt_path = os.path.join(run_dir, "last.pt")
    if resume and os.path.exists(last_ckpt_path):
        ck = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        u, active, targets, history = ck["update"], ck["active"], ck["targets"], ck["history"]
        print(f"resumed {name} at update {u}")

    def progress():
        return u / updates if updates is not None else active / seconds

    eval_every_u = max(1, updates // n_evals) if updates is not None else None
    eval_every_s = seconds / n_evals if seconds is not None else None
    next_eval_s = (len(history) + 1) * eval_every_s if seconds is not None else None
    status, wall0, last_ckpt = "ok", time.time(), time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pf = Prefetcher(train_data, T, B, data_seed=seed, start_update=u)
    loss_ema = None
    try:
        while progress() < 1.0:
            lr_u = lr * lr_factor(progress(), tcfg["schedule"], tcfg["warmup_frac"], tcfg["cooldown_frac"])
            for g in opt.param_groups:
                g["lr"] = lr_u
            tw = time.perf_counter()
            xb = pf.get(u)
            data_wait += time.perf_counter() - tw
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss_acc = torch.zeros((), device=device)
            for i in range(accum):
                x = xb[i * mb:(i + 1) * mb].to(device, non_blocking=True)
                with amp_context(device):
                    logits, _ = model(x[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
                (loss / accum).backward()
                loss_acc += loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            active += time.perf_counter() - t0
            u += 1
            targets += B * T
            train_loss = loss_acc.item() / accum
            if not math.isfinite(train_loss):
                status = "diverged"
                print(f"{name}: non-finite loss at update {u}")
                break
            loss_ema = train_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * train_loss
            due = (u % eval_every_u == 0) if updates is not None else (active >= next_eval_s)
            if due or progress() >= 1.0:
                qb = eval_bpc(model, val_data, T, quick_offs, eval_bs, device)
                history.append({"update": u, "targets": targets, "active_seconds": round(active, 2),
                                "lr": lr_u, "train_bpc_ema": loss_ema / LN2, "quick_val_bpc": qb})
                if seconds is not None:
                    next_eval_s += eval_every_s
                with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
                    f.write(json.dumps(history[-1]) + "\n")
                print(f"{name}  u={u}  t={active:7.1f}s  lr={lr_u:.2e}  train={loss_ema / LN2:.4f}  "
                      f"quick_val={qb:.4f}  ({targets / max(active, 1e-9) / 1e3:.0f}k tgt/s)", flush=True)
                if not math.isfinite(qb):
                    status = "diverged"
                    break
            if updates is not None and time.time() - last_ckpt > ckpt_minutes * 60:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "update": u,
                            "active": active, "targets": targets, "history": history}, last_ckpt_path)
                last_ckpt = time.time()
    finally:
        pf.close()

    final_val = float("nan")
    if status == "ok" and full_val:
        final_val = eval_bpc(model, val_data, T, full_offs, eval_bs, device)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    summary = {
        "run": name, "stage": stage, "arm": arm, "seed": seed, "lr": lr, "budget_kind": budget_kind,
        "budget": budget, "updates": u, "targets": targets, "active_seconds": round(active, 2),
        "wall_seconds": round(time.time() - wall0, 1),
        "targets_per_sec": round(targets / max(active, 1e-9), 1),
        "data_wait_seconds": round(data_wait, 2), "peak_mem_gb": round(peak, 3), "micro_batch": mb,
        "params_nonemb": preport["params_nonemb"], "params_total": preport["params_total"],
        "state_kb_1k": round(state_bytes_per_seq(model, 1024) / 1024, 2),
        "state_kb_16k": round(state_bytes_per_seq(model, 16384) / 1024, 2),
        "train_gflop_per_token": round(train_flops_per_token(model, T) / 1e9, 5),
        "final_val_bpc": final_val, "final_quick_bpc": history[-1]["quick_val_bpc"] if history else float("nan"),
        "status": status,
        "config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
        "git_commit": git_commit(), "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "history": history, "val_windows": len(full_offs),
    }
    torch.save({"model": model.state_dict(), "config": cfg, "summary": summary},
               os.path.join(run_dir, "final.pt"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    if os.path.exists(last_ckpt_path):
        os.remove(last_ckpt_path)
    if write_ledger:
        append_ledger(summary)
    print(f"DONE {name}: status={status} final_val_bpc={final_val:.4f} "
          f"active={active / 60:.1f} min ({summary['targets_per_sec'] / 1e3:.0f}k tgt/s)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--lr", default=None, help="float, or 'auto' = tuned LR from results/lr_choice.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--micro_batch", default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_full_val", action="store_true")
    ap.add_argument("--data_in_ram", action="store_true")
    a = ap.parse_args()
    cfg = load_arm(a.arm)
    lr = resolve_lr(a.arm, cfg, a.lr)
    mb = resolve_micro_batch(a.arm, cfg, a.micro_batch)
    run(a.arm, cfg, lr, a.seed, a.stage, updates=a.updates, seconds=a.seconds, micro_batch=mb,
        device=a.device, force=a.force, resume=a.resume, full_val=not a.no_full_val,
        data_in_ram=a.data_in_ram)


if __name__ == "__main__":
    main()
