# Instructions for Claude Code in this repo

This repo answers one question for a skeptical reviewer: **is the WHAT×WHEN architecture worth more work?**
It does that with a fixed, preregistered Stage A that ends at **checkpoint G1** (a computed GO / NICHE / NO-GO).
Your job is to carry Stage A through, carefully and without changing the design, and stop at G1.

## Read first

1. `PLAN.md` sections 0, 4, 5 and 7 (summary, run sheet, decision rules, change rules). The rest is background.
2. `STATUS.md`: where the work stands. Update it as you go.
3. `DEVIATIONS.md`: anything that differs from the plan, with the reason. Append, never rewrite.

## Golden rules

1. **Follow the run sheet in order** (A0 → A1 → A2 → E1 → A4 → A6 → A5 → A3 → G1). Tick `STATUS.md` when a step's
   *check* passes, not when its command finishes.
2. **Do not change the design** after A1 starts: arms, backbone, budgets, data, evaluation, LR grid, seeds, gate
   thresholds. Allowed changes are listed in PLAN.md §7; log each one in `DEVIATIONS.md` before running.
3. **One GPU job at a time.** Throughput and iso-time results depend on it. No evaluation or benchmark while a
   training run is going.
4. **Every run goes through `python -m ww.queue <stage>`** (or `ww.train` with the same arguments). Never hand-edit
   `results/ledger.csv`. To redo a crashed run use `--force` and note it in `DEVIATIONS.md`.
5. **Tests before results.** All tests must pass before A1 and after any code change. A bug fix gets a test.
6. **The old project is read-only.** `~/Escritorio/THE_RUN_FOR_THE_TROPHIE` (and the offload on HMI-data1) is a
   source of reference numbers, configs and the legacy implementation for A0.5. Nothing here imports it.
7. **Stop at G1.** Write the report, commit, tell the owner. Stage B needs the owner's decision.
8. **Ask the owner** when: the A0 budget estimate exceeds 36 GPU-hours; the anchor run misses 1.970 ± 0.05 and you
   cannot explain why; an arm diverges at every LR; anything would require breaking rule 2.

## Commands

```bash
pip install -r requirements.txt                 # torch with CUDA, flash-linear-attention, numpy, matplotlib
python -m tests.test_whatwhen                   # WHAT×WHEN core: chunked == sequential, gradients, decode
python -m tests.test_model                      # backbone, causality, KV-cache/state decode parity
python -m tests.test_pipeline                   # end-to-end smoke test on synthetic text (CPU, ~10 min)
python -m tests.test_gate_logic                 # the G1 decision rules on synthetic ledgers
python -m tests.test_gpu                        # GPU + fla checks (A0.4)
python -m ww.arms --check --device cuda         # parameter counts, ±3% check, state sizes
python -m ww.bench train --arms tpp,gdn,ww      # throughput, picks micro-batch -> results/micro_batch.json
python -m ww.queue plan                         # what is done, what is left, GPU-hours estimate
python -m ww.queue A1                           # run a stage (A0 A1 A2 A3 A3b A4 A4b A5 A6)
python -m ww.gate select-lr | early | report    # LR choice after A1, E1 after A2, G1 at the end
python -m ww.plots                              # figures into results/figures/
```

`python -m ww.train --help` runs a single run by hand (same naming and ledger as the queue).

## Map

```
PLAN.md              the preregistered plan (what, why, decision rules)
STATUS.md            checklist + log (you update it)
DEVIATIONS.md        every change from the plan, with reason (you append)
docs/                background: the baseline research summary
ww/whatwhen.py       WHAT×WHEN recurrence: sequential definition, chunkwise-parallel form, decode step, mixer
ww/model.py          shared backbone; mixers: attn, swa, ww, gdn, deltanet (fla); accounting helpers
ww/arms.py           every arm defined once; `--write` regenerates configs/, `--check` checks params
ww/data.py           memmap data, paired data order, validation windows
ww/train.py          one run: WSD schedule, iso-token or iso-time budget, resume, ledger row
ww/evaluate.py       validation BPC, long-context evaluation
ww/bench.py          training throughput, data-loading time, decode speed
ww/queue.py          every Stage A run, idempotent; `plan` prints the budget
ww/gate.py           LR selection, early exit E1, the G1 report
ww/plots.py          figures
configs/*.json       generated arm configs (hashed into each ledger row); paths.json = data files
results/             ledger.csv and the other CSVs, lr_choice.json, GATE_REPORT.md (commit these)
runs/                checkpoints and logs (git-ignored)
```

## When things go wrong

- **`import fla` fails or the GatedDeltaNet/DeltaNet constructor rejects an argument:** adapt `FLAMixer` in
  `ww/model.py` to the installed fla version (same layer, same sizes), re-run `tests.test_gpu`, log it.
- **OOM:** lower the micro-batch (`results/micro_batch.json`); it never changes the update.
- **Slow data** (`data_wait_seconds` in the ledger above ~5% of `active_seconds`, or `ww.bench data` > 20 ms per
  batch): the data disk is probably a hard drive. Warm the page cache once per boot
  (`cat <train file> > /dev/null`) or pass `--data_in_ram` (needs ~12 GB free RAM).
- **NaN during the LR sweep:** expected at the highest LRs; the run is recorded as `diverged` and simply loses.
  NaN at the chosen LR in A2/A4 is not expected: re-run that seed once with `--force`; if it diverges again, log it
  and treat the run as failed (the gate reports missing seeds).
- **A long run was interrupted:** just re-run the stage command. Iso-token runs checkpoint every 20 minutes to
  `runs/<name>/last.pt` and the queue always passes `--resume`. If the interruption came from a bug you then fixed,
  delete that `last.pt` first so the run restarts cleanly. Iso-time runs (≤ 40 min) simply restart.
- **WHAT×WHEN much slower than expected:** first try `chunk` 16 or 64 in `ww/arms.py` (exact same function) and a
  larger micro-batch; check `targets_per_sec` in `results/throughput.csv`. Do not change the model.

## Commit discipline

Run `git pull --rebase` before each stage (an independent code review may push fixes; if a fix changes training or
evaluation, follow PLAN §7). Commit **and push** after each stage: `results/*.csv`, `results/*.json`,
`results/GATE_REPORT.md`, `results/figures/`, `STATUS.md`, `DEVIATIONS.md` (and any code change). Never commit `runs/`
(checkpoints) or data. (Amendment 1, PLAN §10.)
