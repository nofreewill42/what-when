# Status

_Update this file as you go: tick a box when its **check** (PLAN.md §4) passes, add one dated line to the log
for anything worth knowing later. Current step is in bold._

## Stage A

- [x] A0.1 environment (torch 2.8.0+cu128, Triton 3.4, fla-core + flash-linear-attention 0.5.2, RTX 3090)
- [x] A0.2 data paths confirmed (`configs/paths.json`), `ww.bench data` timed (50 ms cold; page cache warmed)
- [x] A0.3 CPU tests pass (`tests.test_whatwhen`, `tests.test_model`, `tests.test_gate_logic`, `tests.test_pipeline`)
- [x] A0.4 GPU tests pass (`tests.test_gpu`), gdn/deltanet params within ±3% (gdn FFN 640, +2.7%)
- [x] A0.5 legacy semantics check: new head recurrence == old one on identical controls (rel 2.5e-7)
- [x] A0.6 throughput measured, micro-batches chosen (`results/throughput.csv`, `results/micro_batch.json`), ww chunk 64
- [x] A0.7 anchor: **1.8147**, not 1.970 ± 0.05 — offset explained (old data order: one book per batch), see log
- [x] A0.8 `ww.queue plan` = 108 GPU-h without A4b (> 36) — owner decided **Amendment 1** (PLAN §10): cut plan below
- [x] Amendment 1 items 1, 2, 4, 5 done (push, anchor note, Q4 at chunk 32, ww in_proj fp32: +3.6% time, noise floor unchanged)
- [ ] **A1** LR sweep done (gdn re-swept with the review's conv init), `results/lr_choice.json` has tpp, gdn, ww (edge rule applied)
- [ ] A3 (Amendment 1): `deltanet`, `ww_v8` at R10 seed 0
- [ ] A2 R10 seed 0 (3 arms) — seeds 1, 2 deferred (Amendment 2)
- [ ] E1 run (informational only: one seed, Amendment 2) — result: ______ — **report to owner**
- [—] A4 R40, A4b — deferred (Amendment 1)
- [—] A6 iso-time — deferred (Amendment 1)
- [ ] A5 long-context + decode on the A2 seed-0 checkpoints, figures made
- [—] other A3 ablations, A3b — deferred (Amendment 1)
- [ ] **G1** (from R10, seed 0) `results/GATE_REPORT.md` written and committed — verdict: ______ — **report to owner, stop**

## Measured so far

| item | value |
|---|---|
| throughput tpp / gdn / ww (targets/s) | 607,200 / 678,379 / 33,008 (ww chunk 64, micro-batch 16; chunk 32: 23,171) |
| projected Stage A GPU-hours | 107.9 without A4b (A1 13.7, A2 16.4, A4 21.9, A6 2.6, A3 53.2); 128.9 with A4b |
| anchor (tpp_legacy u8192) | 1.8147 BPC in 350 s, 1.535M tgt/s (old: 1.9702, 362 s, 1.48M) |
| chosen LR tpp / gdn / ww | |

## Log

- 2026-11-01 — repo set up by Claude (web) from the past chats, Claude Code's inventory of the old project and the
  baseline research report. CPU tests pass (core parity, backbone, decode, end-to-end smoke test). Nothing run on GPU.
- 2026-11-01 — A0 by Claude Code. Tests: all CPU and GPU tests pass (two test changes, see DEVIATIONS).
- 2026-11-01 — A0.5: the old 6-layer phase model's layer 0 (seed-0 checkpoint, real validation text) and the new
  `_features` + `ww_sequential`/`ww_chunked` agree to rel 2.5e-7 on identical controls (beta = 1, Fejér spectrum).
  Same conventions: clock includes the current tick, read after write, query at clock + offset, what-major features.
  Old ticks were pi*sigmoid (never negative, trained min 1e-4) -> softplus kept. Learned old spectra stayed at Fejér
  (0.33/0.46/0.21). Parameterisation/init differences (not semantics): old offset pi*tanh with lag-1 init (offset =
  -tick), new off_bias uniform in [-pi, 0]; old periods 4..4096, new 2pi/omega = pi..2048. Script:
  scratchpad `a05_legacy_check.py` (read-only use of the old project).
- 2026-11-01 — A0.6: ww is 18x slower than tpp per update (1.99 s vs 0.108 s), not ~4x as PLAN §6 predicted.
  ww runs at 1.3 TFLOP/s (tpp 15.6). Profile: DtoD copies 17%, fp32 elementwise ~35%, fp32 GEMMs ~32%.
- 2026-11-01 — A0.7 anchor explained. The harness evaluator is exact: the OLD 1.970 checkpoint scores 1.97019 on
  old book-aligned windows (recorded 1.97020) and 1.9706 on ours. Diagnostics (scratchpad, no ledger rows):
  new harness + old residual init 1.8263; new harness + OLD data order 2.0321; old order + old init 2.0231.
  Cause: the old project's `train.schedule.npy` puts all 64 sequences of every update in ONE book (1,373 of 28,595
  books seen in 8,192 updates). That costs ~0.2 BPC and applies to every number in the old project (Transformer and
  WHAT×WHEN alike). The remaining ~0.05 between the old code and the new harness under the old order is within the
  noise of single-book batches (train BPC swings 1.99 -> 2.06 between the last two evaluations). The new harness's
  uniform random windows are correct; the old absolute numbers are not comparable to Stage A numbers.
- 2026-11-01 — Q4 caution for the owner: analytic ww FLOPs/token depend on the chunk size (chunk overhead dominates:
  the recurrence itself is ~4KV per head). ww/tpp = 1.16 at chunk 32 but 1.51 at chunk 64, which crosses the 1.5
  threshold. Owner to decide how Q4 is measured before G1.
- 2026-11-01 — Amendment 1 recorded (PLAN §10). ww throughput with fp32 in_proj: 31,874 targets/s (2,056 ms/update).
- 2026-11-02 — Review commits 31a35b8 / e2e83e1 pulled during A1. gdn's 5 sweep runs used the old conv init ->
  re-run (DEVIATIONS). tpp best 1e-3 (1.6576), ww best so far 1e-3 (1.7106); old-init gdn best 8e-3 (1.5934, edge).
- 2026-11-02 — Amendment 2 (PLAN §11): A2 seed 0 only; deltanet + ww_v8 kept; then A5 and the gate report.
  Remaining work runs unattended via scratchpad `chain_amend2.sh` (log: runs/chain_amend2.log).
- 2026-11-02 — **Paused by the owner.** Chain stopped; ww A1 lr8e-3 run stopped mid-run (last.pt kept, resumes with
  `python -m ww.queue A1`). Not yet done: tests on the review code, gdn re-sweep, select-lr, A3 deltanet/ww_v8,
  A2 seed 0, A5, gate report. To continue: re-launch scratchpad `chain_amend2.sh` steps (or run them by hand).
