# Status

_Update this file as you go: tick a box when its **check** (PLAN.md §4) passes, add one dated line to the log
for anything worth knowing later. Current step is in bold._

## Stage A

- [ ] **A0.1** environment (torch + CUDA + fla import)
- [ ] A0.2 data paths confirmed (`configs/paths.json`), `ww.bench data` timed
- [ ] A0.3 CPU tests pass (`tests.test_whatwhen`, `tests.test_model`, `tests.test_gate_logic`, `tests.test_pipeline`)
- [ ] A0.4 GPU tests pass (`tests.test_gpu`), gdn/deltanet params within ±3%
- [ ] A0.5 legacy semantics check (recommended): new head recurrence == old one on identical controls
- [ ] A0.6 throughput measured, micro-batches chosen (`results/throughput.csv`, `results/micro_batch.json`)
- [ ] A0.7 anchor run reproduces the old D256/L2 number (1.970 ± 0.05 at u8192)
- [ ] A0.8 `ww.queue plan` ≤ 36 GPU-h (else ask the owner) — **report to owner**
- [ ] A1 LR sweep done, `results/lr_choice.json` has tpp, gdn, ww (edge rule applied)
- [ ] A2 R10 done (3 arms × 3 seeds)
- [ ] E1 early-exit check run — result: ______ — **report to owner**
- [ ] A4 R40 done (3 arms, seed 0)   ·   [ ] A4b optional (tpp, ww seed 1)
- [ ] A6 iso-time done (600 s, 2400 s)
- [ ] A5 long-context + decode evaluation done, figures made
- [ ] A3 ablations done, A3b promotions done, A5 re-run for new checkpoints
- [ ] **G1** `results/GATE_REPORT.md` written and committed — verdict: ______ — **report to owner, stop**

## Measured so far

| item | value |
|---|---|
| throughput tpp / gdn / ww (targets/s) | |
| projected Stage A GPU-hours | |
| anchor (tpp_legacy u8192) | |
| chosen LR tpp / gdn / ww | |

## Log

- 2026-11-01 — repo set up by Claude (web) from the past chats, Claude Code's inventory of the old project and the
  baseline research report. CPU tests pass (core parity, backbone, decode, end-to-end smoke test). Nothing run on GPU.
