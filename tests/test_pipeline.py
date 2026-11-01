"""End-to-end CPU smoke test on synthetic text in temp dirs: train (iso-token and iso-time), idempotent
re-run, long-context eval, decode bench, plots, LR selection and the gate. Run: python -m tests.test_pipeline
"""
import copy
import json
import os
import random
import shutil
import tempfile

TMP = tempfile.mkdtemp(prefix="ww_pipeline_")
os.environ["WW_RESULTS"] = os.path.join(TMP, "results")
os.environ["WW_RUNS"] = os.path.join(TMP, "runs")
os.environ["WW_TRAIN"] = os.path.join(TMP, "train.u8")
os.environ["WW_VAL"] = os.path.join(TMP, "val.u8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ww import bench, evaluate, gate, plots, train  # noqa: E402
from ww.arms import ARMS  # noqa: E402


def synthetic_text(n, seed):
    """A tiny Markov 'language': 200 words, sticky transitions, punctuation and line breaks."""
    rng = random.Random(seed)
    words = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 7)))
             for _ in range(200)]
    nxt = {w: [rng.choice(words) for _ in range(4)] for w in words}
    out, w, col, total = [], words[0], 0, 0
    while total < n:
        w = rng.choice(nxt[w]) if rng.random() < 0.85 else rng.choice(words)
        sep = ". " if rng.random() < 0.08 else " "
        col += len(w) + len(sep)
        if col > 70:
            sep, col = "\n", 0
        out += [w, sep]
        total += len(w) + len(sep)
    return "".join(out)[:n].encode("ascii")


def tiny_cfg(arm):
    cfg = copy.deepcopy({k: ARMS[arm][k] for k in ("model", "train", "notes", "lr_from", "param_match", "gpu_only")})
    m = cfg["model"]
    m.update(d_model=32, ffn_hidden=64)
    m["attn"]["n_heads"] = 2
    m["ww"].update(n_heads=8, chunk=16)
    cfg["train"].update(seq_len=64, targets_per_update=64 * 8, micro_batch=4)
    return cfg


def main():
    text = synthetic_text(1_620_000, 0)  # one "language", split into train and validation
    with open(os.environ["WW_TRAIN"], "wb") as f:
        f.write(text[:1_500_000])
    with open(os.environ["WW_VAL"], "wb") as f:
        f.write(text[1_500_000:])
    kw = dict(device="cpu", quick_windows=16, val_max_windows=300, n_evals=4)

    # LR sweep stand-in (two LRs per arm) and the R10 stand-in
    for arm in ("tpp", "ww"):
        for lr in (1e-3, 4e-3):
            s = train.run(arm, tiny_cfg(arm), lr, 0, "A1_lr", updates=40, **kw)
            assert s["status"] == "ok"
    gate.select_lr()
    choice = json.load(open(os.path.join(os.environ["WW_RESULTS"], "lr_choice.json")))
    assert set(choice) == {"tpp", "ww"}
    summaries = {}
    for arm in ("tpp", "ww"):
        s = train.run(arm, tiny_cfg(arm), choice[arm], 0, "A2_r10", updates=150, **kw)
        summaries[arm] = s
        print(arm, "final val bpc", s["final_val_bpc"], "first quick", s["history"][0]["quick_val_bpc"])
        assert s["final_val_bpc"] < s["history"][0]["quick_val_bpc"] - 0.3, "training did not reduce loss"
        assert s["final_val_bpc"] < 4.0
    again = train.run("ww", tiny_cfg("ww"), choice["ww"], 0, "A2_r10", updates=150, **kw)
    assert again["run"] == summaries["ww"]["run"]  # idempotent: skipped, not re-trained
    rows = gate.rows()
    assert sum(r["stage"] == "A2_r10" for r in rows) == 2

    # iso-time budget
    s = train.run("tpp", tiny_cfg("tpp"), choice["tpp"], 0, "A6_iso", seconds=4, **kw)
    assert s["status"] == "ok" and 4.0 <= s["active_seconds"] < 6.0, s["active_seconds"]

    # paired data order: the same seed gives the same windows regardless of micro-batch
    from ww.data import train_offsets
    assert (train_offsets(10**6, 64, 8, 0, 5) == train_offsets(10**6, 64, 8, 0, 5)).all()
    assert not (train_offsets(10**6, 64, 8, 0, 5) == train_offsets(10**6, 64, 8, 1, 5)).all()

    # long context, decode, plots, gate
    for arm in ("tpp", "ww"):
        run_dir = os.path.join(os.environ["WW_RUNS"], summaries[arm]["run"])
        res = evaluate.longctx(run_dir, [64, 256], max_chars=60_000, device=torch.device("cpu"))
        assert len(res) == 2 and all(np.isfinite(r["bpc_all"]) for r in res)
        dec = bench.bench_decode(run_dir, [64, 256], [1, 2], 8, torch.device("cpu"))
        assert len(dec) == 4
    plots.main()
    figs = os.listdir(os.path.join(os.environ["WW_RESULTS"], "figures"))
    assert {"curve_tokens.png", "curve_time.png", "lr_sweep.png", "longctx.png"} <= set(figs), figs
    M = gate.report()
    assert M["verdict"] == "INCOMPLETE"  # no gdn on CPU
    assert os.path.exists(os.path.join(os.environ["WW_RESULTS"], "GATE_REPORT.md"))
    print("PASS pipeline smoke test in", TMP)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
