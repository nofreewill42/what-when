# WHAT×WHEN: is it worth it?

A quick, preregistered comparison of the **WHAT×WHEN** memory (banks of tiny delta-rule fast-weight memories,
addressed by a content angle and a learned per-head clock) against a **well-tuned Transformer++** and **Gated
DeltaNet**, on PG-19 character-level language modelling, one RTX 3090.

- **The plan:** [`PLAN.md`](PLAN.md): the question, the design, the run sheet, and the decision rules of
  checkpoint **G1** (GO / NICHE / NO-GO), written before any run.
- **Who runs it and how:** [`CLAUDE.md`](CLAUDE.md) (operating rules for Claude Code), [`STATUS.md`](STATUS.md)
  (checklist), [`DEVIATIONS.md`](DEVIATIONS.md) (every change from the plan).
- **The answer, when it exists:** `results/GATE_REPORT.md`.

## The design in one paragraph

Every arm is the same 4-layer, d=256 pre-norm residual stack with SwiGLU FFNs and ~3.2M non-embedding parameters;
only the sequence mixer changes (causal attention / Gated DeltaNet / WHAT×WHEN). Each arm gets the same 5-point LR
sweep, the same data in the same order, and the same evaluation. Loss is compared per training character (the
architecture question: 537M and 2.15B characters) and per second (10 and 40 minutes), plus loss at 1k–16k context,
inference memory and decode speed, and 8 ablations of WHAT×WHEN. About 20 GPU-hours to the verdict.

## Quick start

```bash
pip install -r requirements.txt
python -m tests.test_whatwhen && python -m tests.test_model   # CPU, a minute
python -m tests.test_gpu                                      # GPU + flash-linear-attention
python -m ww.queue plan                                       # what is left and how long it takes
python -m ww.queue A1                                         # then follow PLAN.md section 4
```

## Code

| File | What |
|---|---|
| `ww/whatwhen.py` | The WHAT×WHEN recurrence: token-by-token definition, exact chunkwise-parallel form (only a 25×25 state hand-off per chunk is sequential), decode step, the mixer layer |
| `ww/model.py` | Shared backbone and mixers (attention with RoPE/QK-norm/sliding window/KV cache, WHAT×WHEN, fla Gated DeltaNet/DeltaNet) |
| `ww/arms.py` | All 13 arms defined once → `configs/*.json` |
| `ww/train.py`, `ww/queue.py` | Runs (WSD schedule, iso-token or iso-time budget, paired data order, resumable) and the stage queue |
| `ww/evaluate.py`, `ww/bench.py` | Validation and long-context BPC; throughput and decode benchmarks |
| `ww/gate.py`, `ww/plots.py` | LR selection, early exit, the G1 verdict and report; figures |

The chunkwise form is checked against the definition to 1e-10 in fp64 and ~3e-7 in fp32, forward and backward
(`tests/test_whatwhen.py`).
