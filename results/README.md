# results/

Everything here is written by the code; commit it after each stage. Checkpoints live in `runs/` (not committed).

| File | Written by | Content |
|---|---|---|
| `ledger.csv` | `ww.train` | one row per finished run (the single source of truth for the gate) |
| `throughput.csv` | `ww.bench train` | ms/update, targets/s, peak memory, achieved TFLOP/s per arm and micro-batch |
| `micro_batch.json` | `ww.bench train` | fastest micro-batch that fits, per arm (speed only, never changes the update) |
| `lr_choice.json` | `ww.gate select-lr` | tuned LR per main arm (ablations inherit, see `lr_from` in configs) |
| `longctx.csv` | `ww.evaluate longctx` | BPC at L = 1k/4k/16k, all positions and positions ≥ 1024 |
| `decode.csv` | `ww.bench decode` | decode tok/s, peak memory, state/KV bytes per sequence |
| `gate.json`, `GATE_REPORT.md` | `ww.gate report` | the G1 metrics and verdict |
| `figures/*.png` | `ww.plots` | curves vs tokens and time, LR sweep, ablations, per-position loss |

## ledger.csv columns

`run` (name = stage-arm-seed-budget-lr), `stage` (A0_anchor, A1_lr, A2_r10, A3_abl, A4_r40, A6_iso), `arm`, `seed`
(also the data-order seed), `lr`, `budget_kind` (updates | seconds), `budget`, `updates`, `targets`,
`active_seconds` (training only; evaluation and data waits excluded), `wall_seconds`, `targets_per_sec`,
`data_wait_seconds`, `peak_mem_gb`, `micro_batch`, `params_nonemb`, `params_total`, `state_kb_1k`, `state_kb_16k`
(inference memory per sequence), `train_gflop_per_token` (analytic), `final_val_bpc` (full validation),
`final_quick_bpc` (512-window slice), `status` (ok | diverged), `config_sha`, `git_commit`, `device`, `torch`,
`timestamp`.
