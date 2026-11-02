# WHAT×WHEN vs a well-tuned Transformer: a quick go/no-go study

**Status:** preregistered plan, written 2026-11-01. Nothing has run in this repo yet.
**Who runs it:** Claude Code on the owner's machine (one RTX 3090), following `CLAUDE.md`.
**Budget:** about 20 GPU-hours to the checkpoint (G1), set by measurement in step A0.

---

## 0. One-screen summary

**The question a reviewer asks:** is the WHAT×WHEN memory worth more work, or not?

**How we answer it, quickly and honestly:**

1. Put WHAT×WHEN into the *same* small Transformer++ backbone as the baseline, so the only difference is the
   sequence mixer: attention vs WHAT×WHEN vs Gated DeltaNet (the best existing delta-rule layer, with mature kernels).
2. Give each arm the *same* tuning budget (a 5-point learning-rate sweep), the *same* data in the *same* order,
   the *same* parameter count (±3%), and the same evaluation.
3. Measure five things: loss per training token (architecture), loss per second (implementation + architecture),
   loss at long context (1k → 16k characters), inference memory and speed, and which parts of WHAT×WHEN matter (ablations).
4. Stop at **checkpoint G1**. `python -m ww.gate report` applies the decision rules written in section 5 *before* any
   run, and prints **GO**, **NICHE** or **NO-GO** with the numbers behind it.

Everything up to G1 is mechanical: fixed arms, fixed budgets, fixed rules, one command per stage.
Nothing after G1 is planned in detail, because what is worth doing depends on the verdict.

| Step | What | GPU time (est.) |
|---|---|---|
| A0 | Setup, GPU tests, throughput benchmark, reproduce one old number | ~1 h |
| A1 | Learning-rate sweep, 3 arms × 5 LRs, 268M characters each | ~2.5 h |
| A2 | **R10 rung**: 3 arms × 3 seeds, 537M characters (~10 min for the Transformer) | ~3 h |
| E1 | Early-exit check (seconds) | – |
| A4 | **R40 rung**: 3 arms × 1 seed, 2.15B characters (~40 min for the Transformer) | ~4 h |
| A6 | **Iso-time**: 3 arms × exactly 10 min and 40 min of training | 2.5 h |
| A5 | Long-context and decode evaluation of the finished checkpoints | ~1 h |
| A3 | 8 WHAT×WHEN ablations + DeltaNet at R10, 1 seed | ~6 h |
| **G1** | **The gate: verdict + report** | – |

---

## 1. What we already know (from the old project, Aug 28 – Sep 12, 2026)

Source: Claude Code's inventory of `~/Escritorio/THE_RUN_FOR_THE_TROPHIE` on 2026-11-01, and the September chats.
All numbers: PG-19 character level (ASCII-128), T=1024, 65,536 targets per update, official validation split, RTX 3090.

| Model | Params | Budget | Val BPC | Active train time |
|---|---|---|---|---|
| Transformer D256/L2, ReLU, untied | 1.64M | u2048 | 2.273 | 91 s |
| Transformer D256/L2 | 1.64M | u8192 | 1.970 | 362 s |
| Transformer D320/L3, SwiGLU | 3.7M | u32768 | 1.715 | 48 min |
| WHAT×WHEN C3 (3 banks, no FFN) | 1.34M | u2048 | 2.444 | 987 s |
| WHAT×WHEN C4 (4 banks) | 1.74M | u2048 | 2.385 | 1334 s |
| WHAT×WHEN W256D4 (best banked) | 1.46M | u8192 | 2.181 | 4088 s |
| Old 6-layer "phase" model (WHAT×WHEN memory + SiLU FFN) | 6.75M | u8192 | 1.889 | ~7× slower than its Transformer |
| Scan-ReLU-Scan (HGRN scans, a later pivot) | 1.38M | u4096 | 2.170 vs 2.103 Transformer | 178 s vs 181 s |

What these numbers say:

- The **banked** WHAT×WHEN models lose to a tuned Transformer *per token*, not just per second (0.12–0.21 BPC behind at
  equal updates). A faster kernel alone does not fix that.
- The **only** WHAT×WHEN model that ever matched a Transformer per token (the 6-layer phase model) had feed-forward
  layers. Adding nonlinear channel mixing helped every time it was tried. The banked models have none.
- The main established ingredient is **read-conditioning**: a memory whose controls depend on the previous memory's
  read (+0.10 to +0.30 BPC, replicated). In a residual stack this happens automatically: layer k+1 reads what layer k
  wrote into the residual stream.

**Caution (added by Amendment 1, 2026-11-01):** every old-project number above comes from a data order in which
all 64 sequences of each update were drawn from *one book* (A0.7: 1,373 of 28,595 books seen in 8,192 updates).
That handicapped the old runs by roughly 0.2 BPC (the old D256/L2 anchor recipe: 1.970 there, 1.815 in this harness),
and the handicap is not necessarily equal for WHAT×WHEN and the Transformer. The old numbers are context only; they
are not comparable with Stage A results, and no conclusion here rests on them.

What was **never** done: tuning the learning rate for either family (3e-4 throughout); comparing in a shared backbone;
a Gated DeltaNet baseline; long-context evaluation; a forget gate; vector payload (V>1) trained for quality; a GPU
chunkwise kernel. This study does the first five in a quick-and-dirty way and replaces the sixth with an exact
chunkwise form in plain PyTorch (`ww/whatwhen.py`).

---

## 2. What a skeptical reviewer asks, and where the answer comes from

| # | Reviewer question | Measured by | Step |
|---|---|---|---|
| Q1 | Per token of training data, at equal parameters, is it as good as a *tuned* Transformer? | ww − tpp, paired, at R10 (3 seeds) and R40 | A2, A4 |
| Q2 | Is it better than the existing delta-rule layer with good kernels, or just a re-skin? | ww − gdn, same | A2, A4 |
| Q3 | Does it gain from context beyond the training length where the Transformer cannot? | gap change from L=1k to 16k | A5 |
| Q4 | Is the speed gap fundamental, or just an immature implementation? | analytic training FLOPs/token ratio, measured throughput | A0, A2 |
| Q5 | Which parts carry the weight (clock, content angle, FFN, heads, forget gate, payload)? | ablations vs ww | A3 |
| – | What does a user see after 10 and 40 minutes on one GPU? | iso-time runs | A6 |
| – | What does inference cost? | state/KV bytes per sequence, decode tok/s | A5 |

Q1–Q4 decide the verdict (section 5). Q5 and the rest explain it.

---

## 3. Design

### 3.1 One backbone, only the mixer changes

Every arm is the same pre-norm residual stack: `x += Mixer(RMSNorm(x)); x += FFN(RMSNorm(x))`, 4 layers, d_model 256,
SwiGLU FFN (hidden 704), untied embeddings, vocab 128, no biases, zero-initialised residual output projections.
This is the "Transformer++" control used by the DeltaNet / GLA / Gated DeltaNet papers; only the mixer is swapped.

| Arm | Mixer (×4 layers) | Non-emb params | State per sequence | Why it is here |
|---|---|---|---|---|
| `tpp` | causal attention, 4 heads, RoPE, QK-norm | 3.214M | KV cache: 4 MB @1k, 64 MB @16k (bf16) | **primary baseline** |
| `gdn` | Gated DeltaNet (flash-linear-attention), 4 heads × 64 | ≈3.2M (A0 matches ±3%) | ≈0.28 MB constant | best delta-rule layer with mature kernels |
| `ww` | WHAT×WHEN, 128 heads/layer, K=25, V=1, β=1 | 3.215M | 54 KB constant | the architecture under test |

The WHAT×WHEN mixer is the original head recurrence (25-dim Fejér key space, learned per-head clock, β=1 delta rule,
read after write) driven by a linear map of the residual stream instead of a per-character table. See
`ww/whatwhen.py` for the exact equations; it is verified against the token-by-token definition to 3e-7 (fp32) and
1e-10 (fp64), forward and backward.

### 3.2 Why size S (≈3.2M non-embedding parameters)

It is the size the old Transformer frontier picked for 10–40 minute budgets (D320/L3, 3.7M), it is large enough for
depth (4 layers) and FFNs, and small enough that the slowest arm finishes R40 overnight. One size for everyone keeps
Stage A cheap. Each arm choosing its own best size per time budget is a Stage B question.

### 3.3 Data and evaluation

- PG-19, character level, the `prepared_v2` ASCII-128 files from the old project (train 11.4B characters).
- **Paired data order:** the windows of update u are a pure function of (seed, u). Every arm with the same seed
  sees the same characters in the same order, independent of micro-batch size.
- **Final metric:** BPC on the *full* validation file, non-overlapping 1024-character windows, every window from an
  empty context. Curves during training use a fixed 512-window slice (0.5M characters).
- One sweep run sees 2.4% of an epoch, R10 4.7%, R40 19%, so no run repeats data in Stage A.

### 3.4 Training recipe (identical for all arms)

AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on matrices (not on embeddings, norms, 1-D parameters), gradient
clip 1.0, 65,536 targets per update (64 × 1024), BF16 autocast with FP32 master weights (the WHAT×WHEN memory itself
runs in FP32), eager PyTorch (no torch.compile for anyone).

**Schedule: warmup-stable-decay.** Linear warmup over the first 2% of the run, constant, then a (1 − sqrt) cooldown to 0
over the last 20%. Progress is measured in updates (iso-token runs) or active seconds (iso-time runs), so the schedule
always ends exactly when the run ends.

### 3.5 Tuning: equal budget for every arm

- **A1:** each main arm gets the same 5-point LR grid {5e-4, 1e-3, 2e-3, 4e-3, 8e-3} at 4,096 updates (268M targets),
  seed 0. If the best LR is at an edge, the grid is extended one step outward and re-checked, at most twice per side
  (`ww.queue` does this automatically). The lowest final validation BPC wins.
- Ablations inherit `ww`'s LR; DeltaNet inherits `gdn`'s.
- Not tuned for anyone: β2, batch size, warmup length, optimizer. That is the same for every arm, and stated in the report.

### 3.6 Budgets

| Rung | Updates | Targets | Transformer time (est.) | Seeds |
|---|---|---|---|---|
| sweep | 4,096 | 268M | ~5 min | 0 |
| **R10** | 8,192 | 537M | ~10 min | 0, 1, 2 |
| **R40** | 32,768 | 2.15B | ~40 min | 0 (seed 1 optional, A4b) |
| **iso-time** | as many as fit | – | exactly 600 s and 2400 s of active training | 0 |

R10/R40 are iso-token: every arm sees exactly the same characters, so the comparison is about the architecture.
Iso-time runs answer the practical question for the implementations as they are today.

### 3.7 Seeds and statistics

Differences are always *paired* (same seed = same data order). The report gives the mean paired difference, its
standard deviation and a 95% t-interval. With 3 seeds that interval is wide (about ±0.05 BPC), so the decision rules
use point estimates with a tie margin of 0.03 BPC (seed noise at this scale is about 0.015–0.02 per run), and every
single-seed result is flagged. That is the "quick" in quick-and-dirty, and the report says so.

### 3.8 Long context and inference (A5)

- **Long context:** the first 4M validation characters, cut into windows of L = 1024, 4096 and 16384. Recurrent arms
  (and the hybrid, whose attention layer is windowed in training too) run straight through each window. The
  Transformer is scored with **strided 1024-character windows**: each block of 256 characters beyond 1024 is predicted
  by a fresh pass over the 1024 characters ending at it, so every character sees 768–1023 characters of context, which
  is exactly its training condition and the best a 1024-trained Transformer can do on a stream. (A single pass with a
  sliding attention window would be unfair to it: the window evicts the first tokens, which Transformers use as
  attention sinks; the CPU smoke test showed that degradation.) Reported: BPC over all positions, BPC over positions
  ≥ 1024, and a per-position curve.
- **Decode:** batch 1 and 8, contexts 1k/4k/16k, eager PyTorch for every arm, tokens/s and peak memory, plus analytic
  state/KV bytes per sequence (full KV cache and 1024-window KV cache). Gated DeltaNet decode is not wired up; its
  state size is analytic.

### 3.9 Deliberately not in Stage A

Size-optimal frontiers, the 2.5 h and 8 h rungs, Muon and other optimizers, β2/batch tuning, enwik8, more seeds,
a fused Triton kernel for WHAT×WHEN, MQAR-style recall probes. None of them changes a go/no-go answer that the
cheap measurements cannot already give; all are listed for Stage B (section 8).

---

## 4. Run sheet (Stage A)

Every command runs from the repo root. Runs are idempotent: a finished run is never repeated, and re-running a stage
command continues where it stopped. Tick each box in `STATUS.md` when its *check* passes.

### A0. Setup and calibration (~1 h; mostly not GPU)

| # | Do | Check |
|---|---|---|
| A0.1 | Environment: `pip install -r requirements.txt` (PyTorch ≥ 2.4 with CUDA, `flash-linear-attention`) | `python -c "import torch, fla; print(torch.cuda.get_device_name())"` |
| A0.2 | Data paths: confirm both files in `configs/paths.json` (the validation file name is a guess) | `python -m ww.bench data` prints a time per batch |
| A0.3 | CPU tests: `python -m tests.test_whatwhen && python -m tests.test_model && python -m tests.test_gate_logic && python -m tests.test_pipeline` | all PASS |
| A0.4 | GPU tests: `python -m tests.test_gpu` | all PASS. If `gdn`/`deltanet` are outside ±3% params, adjust their `ffn_hidden` in `ww/arms.py` (multiples of 64), `python -m ww.arms --write`, re-run |
| A0.5 | Legacy semantics check (recommended, ≤ 1 h). The new mixer was rebuilt from chat notes, not from the old code. Take one bank of an old checkpoint and a batch of text, feed the same raw per-token controls (kx, ky, qx, qy, tick, off, v) to the old head recurrence and to `WhatWhenMixer._features` + `ww_sequential` | outputs match to fp32 precision. Conventions that may differ: clock inclusive of the current tick or not; read after or before the write; sign of the query offset; feature order and weights; tick parameterisation (ours is softplus, always positive; set `"tick_param": "raw"` in `ww/arms.py` only if the old ticks were unconstrained *and* more than 10% of the trained C3 ticks are negative). Where the old convention differs, adopt it, add a test, log it in `DEVIATIONS.md` |
| A0.6 | Throughput: `python -m ww.bench train --arms tpp,gdn,ww,deltanet,ww_noffn,ww_wide,ww_decay,ww_v8,ww_hybrid` | `results/micro_batch.json` and `results/throughput.csv` exist. Optionally try `ww` chunk 16/64 (same math) and keep the fastest |
| A0.7 | Anchor: `python -m ww.queue A0` (old recipe D256/L2 at u8192) | final BPC within ±0.05 of the old **1.970**. If not, find the reason (data file, eval windows, schedule) before A1; a documented, explained offset is acceptable |
| A0.8 | Budget: `python -m ww.queue plan` | prints remaining GPU-hours. **If Stage A (without A4b) exceeds 36 h, stop and ask the owner** with the two cheapest cuts |

Report to the owner after A0: measured throughputs, projected hours, anchor result.

### A1. LR sweep (~2.5 h)

`python -m ww.queue A1` → `python -m ww.gate select-lr` → check `results/lr_choice.json` has `tpp`, `gdn`, `ww`.
If an arm diverges at every LR, that is a harness bug, not a result: stop and fix.

### A2. R10 rung (~3 h)

`python -m ww.queue A2`, then **E1:** `python -m ww.gate early`.
If E1 says EARLY NO-GO (ww behind tpp by > 0.20 and behind gdn by > 0.15, consistently across seeds), skip A4, A6
and A3, run A5 on the A2 checkpoints, then G1. Otherwise continue.

### A4. R40 rung (~4 h, the WHAT×WHEN run is the long pole)

`python -m ww.queue A4`. Optional, if the owner wants more confidence and time allows: `python -m ww.queue A4b`
(seed 1 for tpp and ww).

### A6. Iso-time (2.5 h)

`python -m ww.queue A6` (600 s and 2400 s, three arms).

### A5. Evaluation (~1 h)

`python -m ww.queue A5` (long context for the R10/R40 seed-0 checkpoints and some ablations; decode for tpp, ww,
ww_hybrid). Then `python -m ww.plots`.

### A3. Ablations (~6 h)

`python -m ww.queue A3`, then `python -m ww.queue A3b` (seed 1 for every ablation that beat ww by ≥ 0.03), then
`python -m ww.queue A5` again (picks up the new long-context runs).

| Ablation | Switch | Question |
|---|---|---|
| `ww_nowhen` | clock off (content-only addressing) | does the WHEN clock matter? |
| `ww_constclock` | tick = learned per-head constant (clock = position × ω) | does a *token-driven* clock beat a fixed-rate one? |
| `ww_nowhat` | content angle off (time-only addressing) | does the WHAT angle matter? |
| `ww_noffn` | no FFN, 384 heads/layer (≈ the old banked design) | was the missing FFN the problem? |
| `ww_wide` | 256 heads, FFN 384 | more memory or more FFN at fixed params? |
| `ww_decay` | learned forget gate α_t | the one lever never tested (+4% params) |
| `ww_v8` | V=8 payload, 48 heads | tensor-core-friendlier shape: quality cost? |
| `ww_hybrid` | layer 3 = sliding-window attention (256) | does a little local attention close the gap? |
| `deltanet` | fla DeltaNet (ungated) | WHAT×WHEN's closest published relative |

### G1. The checkpoint

`python -m ww.plots && python -m ww.gate report` → `results/GATE_REPORT.md`. Commit `results/` (not `runs/`).
Send the owner the verdict line and the main table. **Stop here.** Stage B starts only on the owner's decision.

---

## 5. Checkpoint G1: the preregistered decision rules

All differences are WHAT×WHEN minus baseline, in BPC (negative = WHAT×WHEN better). They come from the R40 seed-0
pair when it exists, otherwise from the R10 3-seed mean. The thresholds are coded in `ww/gate.py` (`THRESH`).

| Question | Measure | Categories |
|---|---|---|
| **Q1** per token vs Transformer++ | ww − tpp | **STRONG** ≤ +0.03 · **MODERATE** ≤ +0.08 · **WEAK** > +0.08 |
| **Q2** per token vs Gated DeltaNet | ww − gdn | **BETTER** ≤ −0.03 · **TIE** · **WORSE** ≥ +0.03 |
| **Q3** long context | (ww − X) at 16k minus (ww − X) at 1k, for X = tpp and gdn | **ADVANTAGE** if ≤ −0.03 against *both* |
| **Q4** speed headroom | analytic training FLOPs/token, ww ÷ tpp | **PLAUSIBLE** ≤ 1.5 · **HARD** > 1.5 |

**Verdict:**

- **GO** = Q1 ∈ {STRONG, MODERATE} **and** Q2 ∈ {BETTER, TIE} **and** Q4 = PLAUSIBLE.
  Competitive per token, not a re-skin of Gated DeltaNet, and the speed gap is an implementation gap.
  → worth a fused kernel and Stage B.
- **NICHE** = not GO, but at least one of: Q2 = BETTER; Q3 = ADVANTAGE; Q1 = STRONG with Q4 = HARD;
  the hybrid arm beats tpp by ≥ 0.03 at R10.
  → worth pursuing only along that angle (better delta-rule addressing, long-context/constant-state inference,
  or a hybrid), not as a general Transformer replacement.
- **NO-GO** = everything else. → write up the negative result; stop investing in this form.

**Flags the report adds (they inform, they do not change the verdict):** a per-token gap that shrank by ≥ 0.04 from R10
to R40 ("run the 2.5 h rung before a final no-go"); any decision resting on a single seed; any arm whose LR sweep
optimum sat at the grid edge.

**E1 early exit (after A2):** ww − tpp > +0.20 *and* ww − gdn > +0.15 (means over 3 seeds, every seed > +0.10)
→ recommend NO-GO without A4/A6/A3. A gap that large does not close with 4× more data at this scale.

---

## 6. Preregistered predictions (Claude, 2026-11-01)

Written before any run, to score our intuition afterwards. Point estimate, 80% interval.

| Quantity | Prediction |
|---|---|
| tpp final BPC at R10 / R40 | 1.88 [1.82, 1.96] / 1.71 [1.66, 1.78] |
| gdn − tpp at R10 | +0.02 [−0.04, +0.08] |
| **ww − tpp at R10** | **+0.05 [−0.03, +0.15]** |
| ww − gdn at R10 | +0.03 [−0.05, +0.10] |
| ww throughput ÷ tpp throughput | 0.25 [0.12, 0.45] |
| ww gap change 1k→16k vs tpp | −0.03 [−0.08, 0.00] |
| ablations (Δ vs ww) | nowhen +0.08 · nowhat +0.15 · constclock +0.02 · noffn +0.15 · wide −0.01 · decay −0.02 · v8 +0.01 · hybrid −0.04 |
| verdict | GO 30% · NICHE 35% · NO-GO 35% |

---

## 7. Rules for changing anything

**Allowed without asking** (log in `DEVIATIONS.md`): bug fixes that keep all tests passing (add a test for the bug);
micro-batch size; the WHAT×WHEN `chunk` size (it does not change the math); adapting the fla wrapper to the installed
fla API; adjusting gdn/deltanet `ffn_hidden` to reach ±3% parameters; data paths; re-running a run that crashed for
an infrastructure reason (use `--force`, note it).

**Not allowed without the owner** after A1 has started: thresholds, arms, the backbone, budgets, the data, the
evaluation protocol, the LR grid, the seeds. If one of them turns out to be wrong, finish Stage A as planned and say
so in the report; a second, clearly labelled analysis is fine, a silently changed one is not.

If a harness bug is found after runs exist: fix it, add a test, re-run every affected run, and record which ledger rows
were superseded.

---

## 8. Threats to validity (and what Stage A does about them)

| Threat | Mitigation in Stage A |
|---|---|
| Under-tuned baseline | Same LR sweep for all arms; tuned Transformer++ recipe (QK-norm, SwiGLU, WSD); the anchor run shows the old untuned recipe for contrast |
| Tuning asymmetry | Identical grid and budget per main arm; ablations do not get their own sweep (stated) |
| Immature WHAT×WHEN kernel | Per-token comparison is the architecture verdict; throughput and analytic FLOPs reported separately |
| Seed noise (~0.015–0.02 BPC) | Paired seeds, 3 at R10, tie margin 0.03, single-seed results flagged |
| Schedule mismatch | WSD sized to every run; no ranking of intermediate checkpoints |
| Different data or eval per arm | One data pipeline, one data order per seed, one evaluation function |
| Long-context cherry-picking | Lengths fixed in advance; the Transformer gets strided 1024-windows (its best case); per-position curves over the same 4M characters |
| Memory claims from formulas | Measured peak memory and tok/s next to the formulas; windowed KV cache reported too |
| Parameter mismatch | ±3% non-embedding parameters, checked by `python -m ww.arms --check` |
| One size, one dataset | Stated as the main limitation; Stage B adds the size frontier and enwik8 |

---

## 9. After G1 (Stage B, only if the verdict asks for it)

- **GO:** a fused Triton chunk kernel for WHAT×WHEN (Fejér closed form for the C×C blocks, no 25-dim feature
  materialisation, deterministic table reduction), then the full ladder 10 min → 40 min → 2.5 h → 8 h where each
  arm picks its best size per rung, Muon as a challenger optimizer, β2 and batch tuned, 2–3 seeds at the large rungs,
  enwik8 as a secondary benchmark, MQAR recall probes, and the paper ("loss vs training time, and inference cost").
- **NICHE:** only the experiments for that angle (for example 16k–64k context and batched decode memory for the
  constant-state angle, or a WHAT×WHEN-addressed Gated DeltaNet for the addressing angle).
- **NO-GO:** a two-page note with the G1 tables; archive.

The research report that chose these baselines (Transformer++ with WSD and equal-budget tuning as the primary
baseline, Gated DeltaNet as the recurrent reference, mixer-swap design, WSD cooldown 20%) is summarised in
`docs/BASELINE_RESEARCH.md`.

---

## 10. Amendment 1 (before A1, 2026-11-01, owner decision after A0)

Made before any WHAT×WHEN-versus-baseline result existed, using only the A0 calibration (STATUS.md). Each item is
also logged in DEVIATIONS.md. The §6 predictions are unchanged.

1. **Git:** commit **and push** after every stage; `git pull --rebase` before each stage (an independent code review
   may push fixes; a fix that changes training or evaluation follows §7).
2. **Anchor (A0.7):** 1.8147 vs the old 1.970 is accepted as an explained offset (old one-book-per-batch data order;
   see the caution in §1 and DEVIATIONS.md).
3. **Budget (A0.8 measured 108 GPU-h > 36):** Stage A is cut to
   A1 → (`deltanet` and `ww_v8` at R10 seed 0, stage `A3_abl`) → A2 → E1 → A5 on the A2 seed-0 checkpoints →
   `python -m ww.gate report` from R10 → **stop and report, whatever E1 says.**
   Deferred (not cancelled): A4, A4b, A6, the other A3 ablations, A3b, the fused kernel. G1 is therefore decided on
   R10 (3 seeds), and the report's Q2/Q1 basis says "R10".
4. **Q4:** the analytic FLOP ratio is computed at **chunk 32** (the preregistered setting) by `ww.gate`;
   training keeps chunk 64 as the runtime choice (same function). GATE_REPORT.md shows the ratio at chunks
   16 / 32 / 64 (0.996 / 1.156 / 1.510) next to the measured throughput ratio.
5. **WHAT×WHEN control projection in fp32:** `in_proj` runs with autocast off in fp32, like the memory core.
   (Result of the check: tests pass, ww 3.6% slower; the bf16 noise floor did *not* drop — see DEVIATIONS.md.)
6. **Reporting:** after A1, after A2/E1 and at the end, push and send the owner each arm's best LR and BPC, grid-edge
   flags, hours left, and finally the verdict.

## 11. Amendment 2 (during A1, 2026-11-02, owner decision)

Made after the A1 sweep points for tpp, gdn and ww (lr ≤ 4e-3) were known; no R10 WHAT×WHEN-versus-baseline result
existed. The owner saw the A1 numbers (ww − tpp ≈ +0.05, ww − gdn ≈ +0.12 at u4096), judged that GO is unlikely and
that Q3 (long context) is the open question, and cut Stage A further:

1. **A2 runs seed 0 only** (tpp, gdn, ww at R10). Seeds 1 and 2 are deferred (not cancelled).
2. Kept: A1 (finished, with the gdn re-sweep below) → `deltanet` and `ww_v8` at R10 seed 0 (A3_abl) → A2 seed 0 →
   A5 on those checkpoints → `ww.gate early` (informational) and `ww.gate report` → stop and report.
3. Consequence, accepted in advance: Q1/Q2 rest on **one seed**; E1's "every seed" rule cannot be applied; the
   report must say so. PLAN §5 already treats a decision resting on a single seed as provisional.
4. **gdn re-sweep (review finding 10, `docs/REVIEW_2026-11-01.md`):** the review commit `31a35b8`, pushed while
   A1 was running, gives fla's short convolutions N(0, 0.02). The 5 gdn A1 runs had used PyTorch's default init,
   so gdn's sweep is re-run with `--force` before `select-lr`. tpp and ww have no convolutions: unaffected.
