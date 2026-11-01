"""The G1 decision rules on synthetic ledgers (no training). Run: python -m tests.test_gate_logic"""
import csv
import json
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


def test_review_fixes():
    """E1 needs every seed > +0.10 against both baselines; the report flags an LR optimum at the grid edge,
    missing R40/hybrid, and stale A5 rows; ablation long-context rows are shown (review of 2026-11-01)."""
    root = tempfile.mkdtemp(prefix="ww_gate2_")
    try:
        d = os.path.join(root, "e1_one_gdn_seed_close")
        write_scenario(d, {"tpp": [1.90, 1.91, 1.89], "gdn": [1.92, 1.93, 2.08], "ww": [2.15, 2.16, 2.14]}, {})
        gate.RESULTS = d
        assert gate.early() is False  # ww - gdn = +0.06 on seed 2

        d = os.path.join(root, "flags")
        os.makedirs(d)
        ts = {"tpp": "2026-11-02 10:00:00", "gdn": "2026-11-02 11:00:00", "ww": "2026-11-02 12:00:00",
              "deltanet": "2026-11-02 13:00:00"}
        led = []
        for arm, lrs in (("tpp", {1e-3: 2.0, 2e-3: 1.95, 4e-3: 1.97}), ("ww", {1e-3: 2.1, 2e-3: 2.2, 4e-3: 2.3})):
            for lr, v in lrs.items():
                led.append(dict(run=f"A1_lr-{arm}-{lr}", stage="A1_lr", arm=arm, seed=0, lr=lr, budget=4096,
                                status="ok", final_val_bpc=v))
        for arm, vals in (("tpp", [1.80, 1.81, 1.79]), ("gdn", [1.82, 1.83, 1.81]), ("ww", [1.84, 1.85, 1.83])):
            for s, v in enumerate(vals):
                led.append(dict(run=f"A2_r10-{arm}-s{s}", stage="A2_r10", arm=arm, seed=s, lr=1e-3, budget=8192,
                                status="ok", final_val_bpc=v, timestamp=ts[arm] if s == 0 else ""))
        led.append(dict(run="A3_abl-deltanet-s0", stage="A3_abl", arm="deltanet", seed=0, lr=1e-3, budget=8192,
                        status="ok", final_val_bpc=1.86, timestamp=ts["deltanet"]))
        with open(os.path.join(d, "ledger.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            w.writeheader()
            for r in led:
                w.writerow({k: r.get(k, "") for k in LEDGER_FIELDS})
        lc = []
        for arm, run, stage, t, by_L in (
                ("tpp", "A2_r10-tpp-s0", "A2_r10", ts["tpp"], {1024: 1.80, 16384: 1.78}),
                ("gdn", "A2_r10-gdn-s0", "A2_r10", ts["gdn"], {1024: 1.82, 16384: 1.79}),
                ("ww", "A2_r10-ww-s0", "A2_r10", "2026-11-01 09:00:00", {1024: 9.0, 16384: 9.0}),  # stale
                ("ww", "A2_r10-ww-s0", "A2_r10", ts["ww"], {1024: 1.84, 16384: 1.77}),
                ("deltanet", "A3_abl-deltanet-s0", "A3_abl", ts["deltanet"], {1024: 1.86, 16384: 1.85})):
            for L, v in by_L.items():
                lc.append(dict(run=run, arm=arm, seed=0, stage=stage, trained_at=t, L=L, n_windows=1, bpc_all=v,
                               bpc_pos_ge_train_len=v, bpc_last_1024=v))
        with open(os.path.join(d, "longctx.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lc[0]))
            w.writeheader()
            w.writerows(lc)
        gate.RESULTS = d
        tab, stage = gate.longctx_table()
        assert stage == "A2_r10" and tab["ww"][1024]["all"] == 1.84 and "deltanet" in tab, tab
        M = gate.report()
        flags = " ".join(M["flags"])
        assert "LR: the best A1 learning rate for ww" in flags and "for tpp" not in flags, flags
        assert "No R40 runs" in flags and "ww_hybrid was not run" in flags, flags
        assert "predate" not in flags, flags  # the stale ww rows were superseded by a current evaluation
        with open(os.path.join(d, "longctx.csv"), "a", newline="") as f:  # a run re-trained after its A5
            csv.writer(f).writerow(["A3_abl-deltanet-s0", "deltanet", 0, "A3_abl", "2026-11-01 08:00:00", 4096,
                                    1, 1.9, 1.9, 1.9])
        with open(os.path.join(d, "ledger.csv"), "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LEDGER_FIELDS).writerow(
                {k: v for k, v in dict(run="A3_abl-ww_v8-s0", stage="A3_abl", arm="ww_v8", seed=0, lr=1e-3,
                                       budget=8192, status="ok", final_val_bpc=1.83,
                                       timestamp="2026-11-02 14:00:00").items()})
        with open(os.path.join(d, "longctx.csv"), "a", newline="") as f:
            csv.writer(f).writerow(["A3_abl-ww_v8-s0", "ww_v8", 0, "A3_abl", "2026-11-01 07:00:00", 1024,
                                    1, 1.8, 1.8, 1.8])
        M = gate.report()
        assert any("A3_abl-ww_v8-s0" in f and "predate" in f for f in M["flags"]), M["flags"]
        # a sweep point re-trained with --force supersedes its earlier ledger row everywhere
        with open(os.path.join(d, "ledger.csv"), "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LEDGER_FIELDS).writerow(
                {k: v for k, v in dict(run="A1_lr-ww-0.001", stage="A1_lr", arm="ww", seed=0, lr=1e-3, budget=4096,
                                       status="ok", final_val_bpc=2.5).items()})
        gate.select_lr()
        with open(os.path.join(d, "lr_choice.json")) as f:
            assert json.load(f)["ww"] == 2e-3
        assert "ww" not in gate.lr_edges()
        from ww import queue
        queue.RESULTS = d
        led = [r for r in queue.ledger_rows() if r["run"] == "A1_lr-ww-0.001"]
        assert len(led) == 1 and float(led[0]["final_val_bpc"]) == 2.5, led
        print("PASS gate review fixes")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    root = tempfile.mkdtemp(prefix="ww_gate_")
    try:
        base10 = {"tpp": [1.90, 1.91, 1.89], "gdn": [1.92, 1.93, 1.91]}

        d = os.path.join(root, "go")
        write_scenario(d, {**base10, "ww": [1.93, 1.94, 1.92]}, {"tpp": 1.72, "gdn": 1.74, "ww": 1.74})
        M = verdict(d)
        assert M["verdict"] == "GO" and M["Q1"] == "STRONG" and M["Q2"] == "TIE", M
        # Amendment 1 item 4: Q4 uses the analytic ratio at chunk 32, not the ledger's (runtime chunk) GFLOP
        assert abs(M["flop_ratio"] - gate.analytic_flop_ratio(32)) < 1e-12 and M["Q4"] == "PLAUSIBLE", M
        assert set(M["flop_ratio_by_chunk"]) == {16, 32, 64}
        assert M["flop_ratio_by_chunk"][16] < M["flop_ratio_by_chunk"][32] < M["flop_ratio_by_chunk"][64]

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
    test_review_fixes()
