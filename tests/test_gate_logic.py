"""The G1 decision rules on synthetic ledgers (no training). Run: python -m tests.test_gate_logic"""
import csv
import os
import shutil
import tempfile

from ww import gate
from ww.train import LEDGER_FIELDS

GFLOP = {"tpp": 0.0257, "gdn": 0.0230, "ww": 0.0296}


def write_scenario(d, r10, r40, lc=None, extra=()):
    """r10: {arm: [bpc seed0, seed1, seed2]}, r40: {arm: bpc seed0}, lc: {arm: {L: bpc_all}}."""
    os.makedirs(d, exist_ok=True)
    rows = []

    def row(stage, arm, seed, budget, val):
        rows.append({"run": f"{stage}-{arm}-s{seed}", "stage": stage, "arm": arm, "seed": seed, "lr": 1e-3,
                     "budget_kind": "updates", "budget": budget, "status": "ok", "final_val_bpc": val,
                     "targets_per_sec": {"tpp": 9e5, "gdn": 6e5}.get(arm, 2.5e5),
                     "train_gflop_per_token": GFLOP.get(arm, 0.03), "state_kb_16k": 54, "active_seconds": 600})

    for arm, vals in r10.items():
        for s, v in enumerate(vals):
            row("A2_r10", arm, s, 8192, v)
    for arm, v in r40.items():
        row("A4_r40", arm, 0, 32768, v)
    for stage, arm, seed, val in extra:
        row(stage, arm, seed, 8192, val)
    with open(os.path.join(d, "ledger.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LEDGER_FIELDS})
    if lc:
        with open(os.path.join(d, "longctx.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", "arm", "seed", "stage", "L", "n_windows", "bpc_all",
                                              "bpc_pos_ge_train_len", "bpc_last_1024"])
            w.writeheader()
            for arm, by_L in lc.items():
                for L, v in by_L.items():
                    w.writerow({"run": f"A4_r40-{arm}-s0", "arm": arm, "seed": 0, "stage": "A4_r40", "L": L,
                                "n_windows": 1, "bpc_all": v, "bpc_pos_ge_train_len": v, "bpc_last_1024": v})


def verdict(d):
    gate.RESULTS = d
    return gate.report()


def main():
    root = tempfile.mkdtemp(prefix="ww_gate_")
    try:
        base10 = {"tpp": [1.90, 1.91, 1.89], "gdn": [1.92, 1.93, 1.91]}

        d = os.path.join(root, "go")
        write_scenario(d, {**base10, "ww": [1.93, 1.94, 1.92]}, {"tpp": 1.72, "gdn": 1.74, "ww": 1.74})
        M = verdict(d)
        assert M["verdict"] == "GO" and M["Q1"] == "STRONG" and M["Q2"] == "TIE", M

        d = os.path.join(root, "nogo")
        write_scenario(d, {**base10, "ww": [2.05, 2.06, 2.04]}, {"tpp": 1.72, "gdn": 1.74, "ww": 1.87})
        M = verdict(d)
        assert M["verdict"] == "NO-GO" and M["Q1"] == "WEAK" and M["Q2"] == "WORSE", M

        d = os.path.join(root, "niche_lc")
        lc = {"tpp": {1024: 1.80, 16384: 1.78}, "gdn": {1024: 1.82, 16384: 1.79}, "ww": {1024: 1.90, 16384: 1.80}}
        write_scenario(d, {**base10, "ww": [2.02, 2.03, 2.01]}, {"tpp": 1.72, "gdn": 1.74, "ww": 1.84}, lc)
        M = verdict(d)
        assert M["verdict"] == "NICHE" and M["Q3"] == "ADVANTAGE", M

        d = os.path.join(root, "niche_vs_gdn")
        write_scenario(d, {"tpp": [1.80, 1.81, 1.79], "gdn": [2.00, 2.01, 1.99], "ww": [1.95, 1.96, 1.94]},
                       {"tpp": 1.62, "gdn": 1.82, "ww": 1.76})
        M = verdict(d)
        assert M["verdict"] == "NICHE" and M["Q2"] == "BETTER" and M["Q1"] == "WEAK", M

        d = os.path.join(root, "trend_flag")
        write_scenario(d, {**base10, "ww": [2.06, 2.07, 2.05]}, {"tpp": 1.72, "gdn": 1.74, "ww": 1.82})
        M = verdict(d)
        assert any("TREND" in f for f in M["flags"]), M["flags"]

        d = os.path.join(root, "early")
        write_scenario(d, {**base10, "ww": [2.15, 2.16, 2.14]}, {})
        gate.RESULTS = d
        assert gate.early() is True

        d = os.path.join(root, "incomplete")
        write_scenario(d, {"tpp": [1.9], "ww": [2.0]}, {})
        assert verdict(d)["verdict"] == "INCOMPLETE"
        print("PASS gate decision rules")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
