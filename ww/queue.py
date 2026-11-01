"""The run queue: every run of Stage A, defined once. Idempotent and resumable.

  python -m ww.queue plan            # list every stage, its runs, done/remaining, and GPU-hour estimate
  python -m ww.queue A1              # run (or continue) a stage; finished runs are skipped
  python -m ww.queue A2 --dry_run    # print the commands without running them
  python -m ww.queue A3 --only deltanet,ww_v8   # only these arms of the stage
  python -m ww.queue A2 --force ww:1            # re-train one finished run (ARM:SEED or the run name);
                                                # log it in DEVIATIONS.md. The ledger keeps both rows and
                                                # every reader uses the latest one.
  python -m ww.queue A1 --data_in_ram           # passed to ww.train (training file in RAM, ~12 GB)

Stages (see PLAN.md for the why):
  A0  anchor run (tpp_legacy, u8192) - the harness must reproduce an old local number
  A1  LR sweep: tpp, gdn, ww at u4096, seed 0, LR grid {5e-4 .. 8e-3} (+ edge extension rule)
  A2  R10 rung: tpp, gdn, ww at u8192 (537M targets), seeds 0, 1, 2
  A4  R40 rung: tpp, gdn, ww at u32768 (2.15B targets), seed 0     [A4b: seed 1 for tpp and ww]
  A6  iso-time: tpp, gdn, ww for exactly 600 s and 2400 s of training, seed 0
  A3  ablations + DeltaNet at u8192, seed 0 (+ seed 1 for any arm that beats ww by >= 0.03)
  A5  evaluation of finished runs: long context (1k/4k/16k) and decode (see `python -m ww.queue A5`)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

from ww.arms import ABLATIONS, ARMS, MAIN_ARMS, load_arm

from ww.paths import RESULTS, ROOT, RUNS
TARGETS_PER_UPDATE = 65536

LR_GRID = [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]
LR_EXTEND = {"low": [2.5e-4, 1.25e-4], "high": [1.6e-2, 3.2e-2]}
U_SWEEP, U_R10, U_R40 = 4096, 8192, 32768
ISO_SECONDS = [600, 2400]


def ledger_rows():
    path = os.path.join(RESULTS, "ledger.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def finished(stage, arm, seed, budget_kind, budget, lr=None):
    for r in ledger_rows():
        if (r["stage"] == stage and r["arm"] == arm and int(r["seed"]) == seed
                and r["budget_kind"] == budget_kind and float(r["budget"]) == float(budget)
                and (lr is None or abs(float(r["lr"]) / lr - 1) < 1e-6)):
            return r
    return None


def lr_choice():
    path = os.path.join(RESULTS, "lr_choice.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def runs_for(stage: str):
    """List of dicts: arm, seed, lr ('auto' or float), updates or seconds."""
    R = []
    if stage == "A0":
        R.append(dict(arm="tpp_legacy", seed=0, lr=None, updates=U_R10, stage="A0_anchor"))
    elif stage == "A1":
        for arm in MAIN_ARMS:
            for lr in LR_GRID:
                R.append(dict(arm=arm, seed=0, lr=lr, updates=U_SWEEP, stage="A1_lr"))
    elif stage == "A2":
        for seed in (0, 1, 2):
            for arm in MAIN_ARMS:
                R.append(dict(arm=arm, seed=seed, lr="auto", updates=U_R10, stage="A2_r10"))
    elif stage == "A3":
        for arm in ABLATIONS:
            R.append(dict(arm=arm, seed=0, lr="auto", updates=U_R10, stage="A3_abl"))
    elif stage == "A3b":  # promotion: seed 1 for arms that beat ww seed 0 by >= 0.03
        ref = finished("A2_r10", "ww", 0, "updates", U_R10)
        if ref is not None:
            for arm in ABLATIONS:
                r = finished("A3_abl", arm, 0, "updates", U_R10)
                if r and r["status"] == "ok" and float(r["final_val_bpc"]) <= float(ref["final_val_bpc"]) - 0.03:
                    R.append(dict(arm=arm, seed=1, lr="auto", updates=U_R10, stage="A3_abl"))
    elif stage == "A4":
        for arm in MAIN_ARMS:
            R.append(dict(arm=arm, seed=0, lr="auto", updates=U_R40, stage="A4_r40"))
    elif stage == "A4b":
        for arm in ("tpp", "ww"):
            R.append(dict(arm=arm, seed=1, lr="auto", updates=U_R40, stage="A4_r40"))
    elif stage == "A6":
        for sec in ISO_SECONDS:
            for arm in MAIN_ARMS:
                R.append(dict(arm=arm, seed=0, lr="auto", seconds=sec, stage="A6_iso"))
    else:
        raise SystemExit(f"unknown stage {stage}")
    return R


def resolved_lr(spec):
    if spec["lr"] == "auto":
        choice = lr_choice()
        if choice is None:
            return None
        cfg = ARMS[spec["arm"]]
        key = spec["arm"] if spec["arm"] in choice else (cfg.get("lr_from") or spec["arm"])
        return choice.get(key)
    if spec["lr"] is None:
        return ARMS[spec["arm"]]["train"]["lr"]
    return spec["lr"]


def is_done(spec):
    budget_kind = "updates" if "updates" in spec else "seconds"
    budget = spec.get("updates", spec.get("seconds"))
    return finished(spec["stage"], spec["arm"], spec["seed"], budget_kind, budget, resolved_lr(spec))


def run_name(spec):
    """The run name ww.train gives this spec (None until its LR is known)."""
    lr = resolved_lr(spec)
    if lr is None:
        return None
    tag = f"u{spec['updates']}" if "updates" in spec else f"t{int(spec['seconds'])}s"
    return f"{spec['stage']}-{spec['arm']}-s{spec['seed']}-{tag}-lr{float(lr):.1e}"


def is_forced(spec, force):
    return any(f == f"{spec['arm']}:{spec['seed']}" or f == run_name(spec) for f in force)


def command(spec, force=False, data_in_ram=False):
    lr = resolved_lr(spec)
    cmd = [sys.executable, "-m", "ww.train", "--arm", spec["arm"], "--seed", str(spec["seed"]),
           "--stage", spec["stage"], "--micro_batch", "auto", "--resume"]
    if lr is not None:
        cmd += ["--lr", repr(float(lr))]
    cmd += ["--updates", str(spec["updates"])] if "updates" in spec else ["--seconds", str(spec["seconds"])]
    if force:
        cmd.append("--force")
    if data_in_ram:
        cmd.append("--data_in_ram")
    return cmd


def throughput():
    """targets/s per arm: measured from finished runs if any, else from results/throughput.csv."""
    thr = {}
    path = os.path.join(RESULTS, "throughput.csv")
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["fits"] == "True":
                    thr[r["arm"]] = max(thr.get(r["arm"], 0.0), float(r["targets_per_sec"]))
    for r in ledger_rows():
        if r["status"] == "ok" and r["budget_kind"] == "updates":
            thr[r["arm"]] = float(r["targets_per_sec"])
    return thr


def est_hours(spec, thr):
    if "seconds" in spec:
        return spec["seconds"] / 3600 * 1.05
    t = thr.get(spec["arm"]) or thr.get(ARMS[spec["arm"]].get("lr_from") or "")  # ablations ~ their parent
    if t is None:
        return None
    return spec["updates"] * TARGETS_PER_UPDATE / t / 3600 * 1.1  # +10% for evaluation


def lr_sweep_extensions():
    """Edge rule: if an arm's best LR is at the edge of what was tried, add the next point outward
    (at most two extra points per side)."""
    extra = []
    for arm in MAIN_ARMS:
        rows = [r for r in ledger_rows() if r["stage"] == "A1_lr" and r["arm"] == arm]
        ok = [r for r in rows if r["status"] == "ok"]
        if len(rows) < len(LR_GRID) or not ok:
            continue
        tried = sorted(float(r["lr"]) for r in rows)
        best = min(ok, key=lambda r: float(r["final_val_bpc"]))
        b = float(best["lr"])
        if abs(b / tried[0] - 1) < 1e-6:
            nxt = [x for x in LR_EXTEND["low"] if x < tried[0]]
        elif abs(b / tried[-1] - 1) < 1e-6:
            nxt = [x for x in LR_EXTEND["high"] if x > tried[-1]]
        else:
            nxt = []
        if nxt:
            extra.append(dict(arm=arm, seed=0, lr=nxt[0], updates=U_SWEEP, stage="A1_lr"))
    return extra


def run_stage(stage, dry_run=False, only=(), force=(), data_in_ram=False):
    specs = [s for s in runs_for(stage) if not only or s["arm"] in only]
    if only and not specs:
        raise SystemExit(f"--only {','.join(only)}: no run of stage {stage} has these arms")
    pending_force = [s for s in specs if is_forced(s, force)]
    if force and not pending_force:
        raise SystemExit(f"--force {','.join(force)}: no run of stage {stage} matches (use ARM:SEED or the run name)")
    while True:
        todo = [s for s in specs if s in pending_force or not is_done(s)]
        for spec in todo:
            if spec["lr"] == "auto" and resolved_lr(spec) is None:
                raise SystemExit("results/lr_choice.json missing: run `python -m ww.gate select-lr` first")
            cmd = command(spec, force=spec in pending_force, data_in_ram=data_in_ram)
            print(" ".join(cmd), flush=True)
            if not dry_run:
                rc = subprocess.call(cmd, cwd=ROOT)
                if rc != 0:
                    raise SystemExit(f"run failed with exit code {rc}: {' '.join(cmd)}")
        pending_force = []  # a forced run is re-trained once
        if stage != "A1" or dry_run:
            break
        ext = [s for s in lr_sweep_extensions() if not is_done(s) and (not only or s["arm"] in only)]
        if not ext:
            break
        print(f"LR edge rule: adding {[(s['arm'], s['lr']) for s in ext]}")
        specs += ext


def latest_ok(stage, arm, seed=0):
    """The latest ledger row of (stage, arm, seed) if that run finished ok (a --force re-run adds a row)."""
    last = None
    for r in ledger_rows():
        if r["stage"] == stage and r["arm"] == arm and int(r["seed"]) == seed:
            last = r
    return last if last is not None and last["status"] == "ok" else None


def _evaluated(name):
    """{run: {trained_at, ...}} of the checkpoints already evaluated in results/<name>."""
    path = os.path.join(RESULTS, name)
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                out.setdefault(r["run"], set()).add(r.get("trained_at", ""))
    return out


def run_a5(dry_run=False):
    """Evaluate finished checkpoints: long context for the main arms (R10 and R40, seed 0) and the
    clock/memory ablations, decode speed for the longest-trained seed-0 tpp and ww runs (and ww_hybrid,
    ww_v8 if they ran). A checkpoint counts as evaluated only if its rows carry its own training timestamp,
    so a re-trained run (--force) is evaluated again."""
    done_lc, done_dec = _evaluated("longctx.csv"), _evaluated("decode.csv")
    targets = []
    for stage in ("A2_r10", "A4_r40"):
        for arm in MAIN_ARMS:
            targets.append((stage, arm))
    for arm in ("ww_constclock", "ww_nowhen", "ww_decay", "ww_hybrid", "ww_v8", "deltanet"):
        targets.append(("A3_abl", arm))
    cmds = []
    for stage, arm in targets:
        r = latest_ok(stage, arm)
        if r and r["timestamp"] not in done_lc.get(r["run"], set()):
            cmds.append([sys.executable, "-m", "ww.evaluate", "longctx", "--run", os.path.join(RUNS, r["run"])])
    for arm in ("tpp", "ww", "ww_hybrid", "ww_v8"):
        r = None
        for stage in ("A4_r40", "A2_r10", "A3_abl"):
            r = r or latest_ok(stage, arm)
        if r and r["timestamp"] not in done_dec.get(r["run"], set()):
            cmds.append([sys.executable, "-m", "ww.bench", "decode", "--run", os.path.join(RUNS, r["run"])])
    for cmd in cmds:
        print(" ".join(cmd), flush=True)
        if not dry_run:
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                raise SystemExit(f"failed: {' '.join(cmd)}")
    if not cmds:
        print("A5: nothing left to evaluate")


def plan():
    thr = throughput()
    total, unknown = 0.0, []
    print(f"{'stage':6s} {'runs':>5s} {'done':>5s} {'est GPU-h left':>15s}")
    for stage in ("A0", "A1", "A2", "A4", "A4b", "A6", "A3", "A3b"):
        specs = runs_for(stage)
        done = sum(1 for s in specs if is_done(s))
        hours = 0.0
        for s in specs:
            if is_done(s):
                continue
            h = est_hours(s, thr)
            if h is None:
                unknown.append(s["arm"])
            else:
                hours += h
        total += hours
        opt = "  (optional)" if stage in ("A4b",) else ""
        print(f"{stage:6s} {len(specs):5d} {done:5d} {hours:15.1f}{opt}")
    print(f"total remaining (incl. optional A4b): {total:.1f} GPU-h")
    if unknown:
        print(f"no throughput yet for: {sorted(set(unknown))}  (run `python -m ww.bench train` in A0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", help="plan | A0 | A1 | A2 | A3 | A3b | A4 | A4b | A5 | A6")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated arms: run only these arms of the stage")
    ap.add_argument("--force", default="",
                    help="comma-separated ARM:SEED or run names to re-train even if finished (log it)")
    ap.add_argument("--data_in_ram", action="store_true", help="passed to ww.train")
    a = ap.parse_args()
    split = lambda s: tuple(x.strip() for x in s.split(",") if x.strip())
    if a.stage == "plan":
        plan()
    elif a.stage == "A5":
        run_a5(a.dry_run)
    else:
        run_stage(a.stage, a.dry_run, only=split(a.only), force=split(a.force), data_in_ram=a.data_in_ram)


if __name__ == "__main__":
    main()
