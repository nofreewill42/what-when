"""LR selection, the early-exit check, and the preregistered go/no-go gate (checkpoint G1).

  python -m ww.gate select-lr   # after A1: best LR per arm -> results/lr_choice.json
  python -m ww.gate early       # after A2: early-exit check E1
  python -m ww.gate report      # checkpoint G1: results/GATE_REPORT.md + results/gate.json

The thresholds below are the preregistration (PLAN.md section 5). Do not edit them after A1 has started;
if one turns out to be wrong, say so in the report instead (PLAN.md section 7).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time

from ww.arms import ABLATIONS, ARMS, MAIN_ARMS

from ww.paths import RESULTS

THRESH = {
    "tie": 0.03,         # |paired difference| below this is a tie (seed noise ~0.015-0.02 BPC/run)
    "close": 0.08,       # per-token gap up to this counts as "close" to the Transformer
    "lc": 0.03,          # long-context advantage: the gap must shrink by this from 1k to 16k
    "flop_ratio": 1.5,   # speed is an implementation problem only if WW needs <= 1.5x the FLOPs/token
    "early_tpp": 0.20,   # E1: WW behind tpp by more than this at R10 ...
    "early_gdn": 0.15,   # ... and behind gdn by more than this -> recommend stopping early
    "trend": 0.04,       # gap shrinking from R10 to R40 by more than this is flagged
}
T_CRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}


def rows():
    path = os.path.join(RESULTS, "ledger.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            for k in ("lr", "budget", "targets_per_sec", "final_val_bpc", "active_seconds",
                      "train_gflop_per_token", "params_nonemb", "state_kb_1k", "state_kb_16k", "peak_mem_gb"):
                try:
                    r[k] = float(r[k])
                except (ValueError, KeyError):
                    r[k] = float("nan")
            r["seed"] = int(r["seed"])
            out.append(r)
        return out


def bpc(stage, arm, budget=None):
    """{seed: final_val_bpc} for finished runs (the latest row wins if a run was repeated)."""
    out = {}
    for r in rows():
        if r["stage"] == stage and r["arm"] == arm and r["status"] == "ok" and not math.isnan(r["final_val_bpc"]):
            if budget is None or r["budget"] == float(budget):
                out[r["seed"]] = r["final_val_bpc"]
    return out


def paired(a: dict, b: dict):
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    if not d:
        return None
    mean = sum(d) / len(d)
    sd = statistics.stdev(d) if len(d) > 1 else float("nan")
    half = T_CRIT.get(len(d) - 1, 2.0) * sd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"n": len(d), "mean": mean, "sd": sd, "ci95": (mean - half, mean + half), "diffs": d, "seeds": seeds}


def fmt(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def fmt_pair(p):
    if p is None:
        return "missing"
    ci = "" if p["n"] < 2 else f" (95% CI {p['ci95'][0]:+.3f} to {p['ci95'][1]:+.3f})"
    return f"{p['mean']:+.3f} over {p['n']} seed(s){ci}"


# ------------------------------------------------------------------------------------------------
def select_lr():
    choice, lines = {}, []
    for arm in sorted({r["arm"] for r in rows() if r["stage"] == "A1_lr"}):
        rs = sorted([r for r in rows() if r["stage"] == "A1_lr" and r["arm"] == arm], key=lambda r: r["lr"])
        ok = [r for r in rs if r["status"] == "ok" and not math.isnan(r["final_val_bpc"])]
        if not ok:
            lines.append(f"{arm}: no finished run")
            continue
        best = min(ok, key=lambda r: r["final_val_bpc"])
        edge = best["lr"] in (rs[0]["lr"], rs[-1]["lr"])
        choice[arm] = best["lr"]
        cells = "  ".join(f"{r['lr']:.1e}:{fmt(r['final_val_bpc']) if r['status'] == 'ok' else r['status']}" for r in rs)
        lines.append(f"{arm:6s} best {best['lr']:.1e} ({best['final_val_bpc']:.4f})"
                     f"{'  <-- AT EDGE: queue A1 again (edge rule)' if edge else ''}\n       {cells}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "lr_choice.json"), "w") as f:
        json.dump(choice, f, indent=1)
    print("\n".join(lines))
    print(f"wrote results/lr_choice.json: {choice}")
    missing = [a for a in MAIN_ARMS if a not in choice]
    if missing:
        print(f"WARNING: no LR for {missing}")


def early():
    t, g, w = bpc("A2_r10", "tpp"), bpc("A2_r10", "gdn"), bpc("A2_r10", "ww")
    dT, dG = paired(w, t), paired(w, g)
    print(f"R10  ww - tpp: {fmt_pair(dT)}")
    print(f"R10  ww - gdn: {fmt_pair(dG)}")
    if dT is None or dG is None or dT["n"] < 3:
        print("E1 needs all three A2 seeds for ww, tpp and gdn.")
        return None
    stop = (dT["mean"] > THRESH["early_tpp"] and dG["mean"] > THRESH["early_gdn"]
            and all(x > THRESH["early_tpp"] / 2 for x in dT["diffs"]))
    print("E1: EARLY NO-GO recommended (skip A4/A6/A3; run A5 on the A2 runs, then the gate)" if stop
          else "E1: continue with A4")
    return stop


# ------------------------------------------------------------------------------------------------
def longctx_table():
    path = os.path.join(RESULTS, "longctx.csv")
    if not os.path.exists(path):
        return {}, None
    with open(path) as f:
        lc = list(csv.DictReader(f))
    for stage in ("A4_r40", "A2_r10"):  # prefer the longest-trained seed-0 checkpoints
        sel = {}
        for r in lc:
            if r["stage"] == stage and int(r["seed"]) == 0:
                sel.setdefault(r["arm"], {})[int(r["L"])] = {
                    "all": float(r["bpc_all"]), "late": float(r["bpc_pos_ge_train_len"])}
        if all(a in sel for a in MAIN_ARMS):
            return sel, stage
    return {}, None


def lc_change(tab, x, lo=1024, hi=16384):
    try:
        return (tab["ww"][hi]["all"] - tab[x][hi]["all"]) - (tab["ww"][lo]["all"] - tab[x][lo]["all"])
    except KeyError:
        return None


def median_field(stage, arm, field):
    vals = [r[field] for r in rows() if r["stage"] == stage and r["arm"] == arm and r["status"] == "ok"
            and not math.isnan(r[field])]
    return statistics.median(vals) if vals else None


def report():
    M = {}
    r10 = {a: bpc("A2_r10", a) for a in MAIN_ARMS}
    r40 = {a: bpc("A4_r40", a) for a in MAIN_ARMS}
    M["r10_ww_tpp"] = paired(r10["ww"], r10["tpp"])
    M["r10_ww_gdn"] = paired(r10["ww"], r10["gdn"])
    M["r10_gdn_tpp"] = paired(r10["gdn"], r10["tpp"])
    M["r40_ww_tpp"] = paired(r40["ww"], r40["tpp"])
    M["r40_ww_gdn"] = paired(r40["ww"], r40["gdn"])
    thr = {a: median_field("A2_r10", a, "targets_per_sec") for a in MAIN_ARMS + ABLATIONS}
    gfl = {a: median_field("A2_r10", a, "train_gflop_per_token") or median_field("A3_abl", a, "train_gflop_per_token")
           for a in MAIN_ARMS + ABLATIONS}
    iso = {a: {int(r["budget"]): r["final_val_bpc"] for r in rows()
               if r["stage"] == "A6_iso" and r["arm"] == a and r["status"] == "ok"} for a in MAIN_ARMS}
    lc, lc_stage = longctx_table()
    M["lc_vs_tpp"], M["lc_vs_gdn"] = lc_change(lc, "tpp"), lc_change(lc, "gdn")

    # ---- the preregistered decision -----------------------------------------------------------
    reasons, flags = [], []
    q1 = M["r40_ww_tpp"] or M["r10_ww_tpp"]
    q2 = M["r40_ww_gdn"] or M["r10_ww_gdn"]
    basis = "R40" if M["r40_ww_tpp"] else "R10"
    if q1 is None or q2 is None:
        verdict = "INCOMPLETE"
        reasons.append("A2 (and ideally A4) results for ww, tpp and gdn are needed.")
        Q1 = Q2 = Q3 = Q4 = None
    else:
        Q1 = "STRONG" if q1["mean"] <= THRESH["tie"] else ("MODERATE" if q1["mean"] <= THRESH["close"] else "WEAK")
        Q2 = "BETTER" if q2["mean"] <= -THRESH["tie"] else ("WORSE" if q2["mean"] >= THRESH["tie"] else "TIE")
        lc_t, lc_g = M["lc_vs_tpp"], M["lc_vs_gdn"]
        Q3 = None if lc_t is None else ("ADVANTAGE" if (lc_t <= -THRESH["lc"] and lc_g is not None
                                                        and lc_g <= -THRESH["lc"]) else "NONE")
        flop_ratio = (gfl["ww"] / gfl["tpp"]) if gfl.get("ww") and gfl.get("tpp") else None
        Q4 = None if flop_ratio is None else ("PLAUSIBLE" if flop_ratio <= THRESH["flop_ratio"] else "HARD")
        M["flop_ratio"] = flop_ratio
        reasons.append(f"Q1 per-token vs Transformer++ ({basis}): ww - tpp = {q1['mean']:+.3f} -> {Q1}")
        reasons.append(f"Q2 per-token vs Gated DeltaNet ({basis}): ww - gdn = {q2['mean']:+.3f} -> {Q2}")
        reasons.append(f"Q3 long context (gap change 1k->16k): vs tpp {fmt(lc_t, signed=True)}, "
                       f"vs gdn {fmt(lc_g, signed=True)} -> {Q3 or 'missing'}")
        reasons.append(f"Q4 speed headroom: ww/tpp training FLOPs per token = {fmt(flop_ratio, 2)} -> {Q4 or 'missing'}"
                       f"; measured throughput ratio = {fmt((thr['ww'] or 0) / thr['tpp'], 2) if thr.get('tpp') else '–'}")
        hyb = paired(bpc("A3_abl", "ww_hybrid"), r10["tpp"])
        if Q1 in ("STRONG", "MODERATE") and Q2 in ("BETTER", "TIE") and Q4 == "PLAUSIBLE":
            verdict = "GO"
        elif Q2 == "BETTER" or Q3 == "ADVANTAGE" or (Q1 == "STRONG" and Q4 == "HARD") \
                or (hyb is not None and hyb["mean"] <= -THRESH["tie"]):
            verdict = "NICHE"
            if hyb is not None and hyb["mean"] <= -THRESH["tie"]:
                reasons.append(f"hybrid (3 WW + 1 SWA) beats tpp at R10 by {-hyb['mean']:.3f}")
        else:
            verdict = "NO-GO"
        if M["r10_ww_tpp"] and M["r40_ww_tpp"]:
            shrink = M["r10_ww_tpp"]["mean"] - M["r40_ww_tpp"]["mean"]
            M["trend_shrink"] = shrink
            if shrink >= THRESH["trend"]:
                flags.append(f"TREND: the per-token gap to tpp shrank by {shrink:.3f} from R10 to R40. "
                             "Before a final no-go, run tpp and ww for 2.5 h (Stage B rung) to see where it goes.")
        if q1["n"] < 2:
            flags.append(f"{basis} rests on {q1['n']} seed; treat differences below 0.05 as unresolved.")
    M.update({"verdict": verdict, "Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4, "reasons": reasons, "flags": flags})

    # ---- write the report ------------------------------------------------------------------------
    L = []
    L.append("# Gate G1 report: is WHAT×WHEN worth pursuing?\n")
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m ww.gate report` from results/ledger.csv. "
             "Thresholds were preregistered in PLAN.md section 5._\n")
    meaning = {"GO": "worth a real kernel and Stage B (longer rungs, size-optimal frontier, paper).",
               "NICHE": "worth pursuing only along the angle named below, not as a general Transformer replacement.",
               "NO-GO": "not worth further investment in this form; write up the negative result.",
               "INCOMPLETE": "not enough finished runs to decide."}
    L.append(f"## Verdict: **{verdict}** — {meaning[verdict]}\n")
    L += [f"- {r}" for r in reasons]
    L += [f"- **Flag:** {f}" for f in flags]
    L.append("")
    L.append("## Main arms (size S, ~3.2M non-embedding params, same data order, full PG-19 validation)\n")
    L.append("| arm | R10 BPC (mean ± sd, n) | R40 BPC (seed 0) | 10 min BPC | 40 min BPC | train targets/s | "
             "GFLOP/token | state @16k |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in MAIN_ARMS:
        v = list(r10[a].values())
        r10s = f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f} (n={len(v)})" if len(v) > 1 else \
            (f"{v[0]:.4f} (n=1)" if v else "–")
        st = median_field("A2_r10", a, "state_kb_16k")
        L.append(f"| {a} | {r10s} | {fmt(r40[a].get(0), 4)} | {fmt(iso[a].get(600), 4)} | "
                 f"{fmt(iso[a].get(2400), 4)} | {fmt(thr.get(a), 0)} | {fmt(gfl.get(a), 4)} | "
                 f"{fmt(st, 1)} KB |")
    L.append("")
    L.append("## Paired differences (negative = WHAT×WHEN better)\n")
    for key, label in (("r10_ww_tpp", "R10 ww − tpp"), ("r10_ww_gdn", "R10 ww − gdn"),
                       ("r10_gdn_tpp", "R10 gdn − tpp"), ("r40_ww_tpp", "R40 ww − tpp"),
                       ("r40_ww_gdn", "R40 ww − gdn")):
        L.append(f"- {label}: {fmt_pair(M[key])}")
    L.append("")
    L.append("## Ablations at R10 (seed 0, LR inherited from ww; Δ = arm − ww, negative = better than ww)\n")
    ref = r10["ww"].get(0)
    L.append("| arm | BPC | Δ vs ww | targets/s | what it tests |")
    L.append("|---|---|---|---|---|")
    for a in ABLATIONS:
        b = bpc("A3_abl", a).get(0)
        d = (b - ref) if (b is not None and ref is not None) else None
        L.append(f"| {a} | {fmt(b, 4)} | {fmt(d, 3, True)} | {fmt(median_field('A3_abl', a, 'targets_per_sec'), 0)} | "
                 f"{ARMS[a]['notes']} |")
    L.append("\nSingle-seed ablation differences smaller than ±0.03 are within noise.\n")
    L.append(f"## Long context (checkpoints from {lc_stage or '–'}, seed 0; the Transformer is scored with "
             "strided 1024-character windows beyond 1024)\n")
    if lc:
        Ls = sorted({k for a in lc for k in lc[a]})
        L.append("| arm | " + " | ".join(f"BPC @ L={x}" for x in Ls) + f" | BPC at positions ≥1024 (L={Ls[-1]}) |")
        L.append("|---|" + "---|" * (len(Ls) + 1))
        for a in [x for x in MAIN_ARMS + ABLATIONS if x in lc]:
            L.append(f"| {a} | " + " | ".join(fmt(lc[a].get(x, {}).get("all"), 4) for x in Ls) +
                     f" | {fmt(lc[a].get(Ls[-1], {}).get('late'), 4)} |")
        L.append(f"\nGap change 1k→16k (ww − X; negative = ww gains more from context): vs tpp "
                 f"{fmt(M['lc_vs_tpp'], 3, True)}, vs gdn {fmt(M['lc_vs_gdn'], 3, True)}\n")
    else:
        L.append("missing: run A5 (`python -m ww.queue A5` prints the commands)\n")
    dec = os.path.join(RESULTS, "decode.csv")
    if os.path.exists(dec):
        L.append("## Decode (eager PyTorch, batch 1 and 8)\n")
        with open(dec) as f:
            drows = list(csv.DictReader(f))
        L.append("| arm | context | batch | tok/s | peak MB | state KB/seq (as deployed) | KV KB/seq if full cache |")
        L.append("|---|---|---|---|---|---|---|")
        for r in drows:
            L.append(f"| {r['arm']} | {r['context']} | {r['batch']} | {r['tok_per_s']} | {r['peak_mem_mb']} | "
                     f"{r['state_kb_per_seq_windowed']} | {r['state_kb_per_seq_fullkv']} |")
        L.append("")
    L.append("## LR sweep (A1, u4096, seed 0)\n")
    for a in MAIN_ARMS:
        rs = sorted([r for r in rows() if r["stage"] == "A1_lr" and r["arm"] == a], key=lambda r: r["lr"])
        L.append(f"- {a}: " + ", ".join(f"{r['lr']:.1e} → {fmt(r['final_val_bpc'], 4) if r['status'] == 'ok' else r['status']}"
                                        for r in rs))
    L.append("")
    figs = os.path.join(RESULTS, "figures")
    if os.path.isdir(figs):
        L.append("## Figures\n")
        for fn in sorted(os.listdir(figs)):
            if fn.endswith(".png"):
                L.append(f"![{fn}](figures/{fn})")
        L.append("")
    L.append("## Read before trusting this\n")
    L.append("- One fixed size (S) for every arm: this says nothing yet about which architecture wins when each "
             "picks its own best size for a time budget (Stage B).")
    L.append("- LR was tuned at u4096 and transferred; β2, batch size, warmup and the optimizer were not tuned.")
    L.append("- Wall-clock compares implementations, not just architectures: tpp uses fused SDPA, gdn uses fla "
             "Triton kernels, ww uses plain PyTorch ops. Per-token numbers are the architecture comparison.")
    L.append("- Ablation LRs are inherited from ww; any ablation that beat ww got a second seed (A3b).")
    with open(os.path.join(RESULTS, "GATE_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and math.isnan(o):
            return None
        return o

    with open(os.path.join(RESULTS, "gate.json"), "w") as f:
        json.dump(clean(M), f, indent=1)
    print("\n".join(L[:12]))
    print("\nwrote results/GATE_REPORT.md and results/gate.json")
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["select-lr", "early", "report"])
    a = ap.parse_args()
    {"select-lr": select_lr, "early": early, "report": report}[a.what]()


if __name__ == "__main__":
    main()
