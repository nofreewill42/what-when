# Deviations from PLAN.md

Append one entry per change, **before** running anything that depends on it. Never rewrite old entries.

Format:

```
## YYYY-MM-DD — short title
- What changed:
- Why:
- Allowed by: PLAN.md §7 (which item) / owner (who, when)
- Affected runs (ledger rows superseded, if any):
```

(no entries yet)

## 2026-11-01 — environment: installed the `flash-linear-attention` layers package
- What changed: the machine had `fla-core 0.5.2` (kernels only, no `fla.layers`). Installed
  `flash-linear-attention==0.5.2` with `pip install --no-deps` (a dry run showed it would add only this package;
  torch 2.8.0+cu128 / Triton 3.4.0 unchanged). No change to `FLAMixer` was needed.
- Why: A0.1/A0.4 need `fla.layers.GatedDeltaNet` / `DeltaNet`.
- Allowed by: PLAN.md §4 A0.1 (environment), CLAUDE.md ("pip install -r requirements.txt").
- Affected runs: none (before any run).

## 2026-11-01 — gdn FFN hidden 704 -> 640 (parameter match)
- What changed: `ww/arms.py` gdn `ffn_hidden` 640. Non-embedding params 3,496,480 (+8.8%) -> 3,299,872 (+2.7% vs tpp).
  576 would give -3.4%, so 640 is the only multiple of 64 inside ±3%.
- Why: fla GatedDeltaNet (4 heads × 64, expand_v 1, short conv, output gate) has more mixer parameters than attention.
- Allowed by: PLAN.md §7 ("adjusting gdn/deltanet ffn_hidden to reach ±3% parameters").
- Affected runs: none.

## 2026-11-01 — tpp_legacy: the exact old FFN (SwiGLU 672), as the arm's own note required
- What changed: `tpp_legacy` FFN `relu`/1024 -> `swiglu`/672 (qk_norm stays off). The 1.970 anchor is the old
  `budget_grid_v1` arm `tr_d256_l2_swiglu_h672_u08192` (seed 0, 1.97020 BPC at u8192, 1,623,296 params; config in
  `THE_RUN_FOR_THE_TROPHIE/runs/pilots/whatwhen_transformer_budget_scaling_grid_v1_20260909_02/`). The new arm now has
  exactly 1,623,296 parameters. (The old ReLU-1024 D256/L2 arm has no seed-0 u8192 number; seed 1 was 2.0128.)
- Remaining known differences from the old anchor, kept on purpose because they are properties of the new harness
  that the anchor is meant to test: residual output projections zero-initialised (old: N(0, 0.02)/sqrt(2L));
  weight decay 0.01 on matrices except the embedding (old: all parameters); training windows at uniformly random
  offsets (old: book-ordered schedule); validation = non-overlapping 1024-windows of the validation stream
  (old: book-aligned blocks, 16,934 of them); attention via torch SDPA (old: xFormers).
- Why: the arm note says "A0 must copy the exact old FFN kind/width from the old project before running".
- Allowed by: PLAN.md §4 A0.7 (anchor setup, before A1).
- Affected runs: none.

## 2026-11-01 — test fix: bf16 decode-parity check compares with the model's own noise floor
- What changed: `tests/test_gpu.py::test_bf16_decode_close_to_full_forward` now requires fp32 decode == full forward
  (< 1e-3, measured 8.9e-7 tpp / 1.9e-4 ww) and bf16 decode within max(0.1, 2 × the bf16 noise floor), where the
  noise floor is the change from re-running the identical full forward on a batch slice.
- Why: the old fixed 0.1 bound failed for ww at 0.107, but re-running ww's own full forward on one batch row already
  moves its logits by 0.135 in bf16 (tpp: 0.008). Decode is correct (fp32 parity); WHAT×WHEN amplifies bf16 rounding
  of its control projection because (kx, ky) and (qx, qy) are normalised to angles. No model change.
- Allowed by: PLAN.md §7 (bug fix that keeps all tests passing; the fix is in the test).
- Affected runs: none.

## 2026-11-01 — bug fix: analytic WHAT×WHEN FLOPs (gate Q4) missed two terms
- What changed: `train_flops_per_token` now uses `ww_core_flops_per_head` in `ww/model.py`, which adds the
  `u = U0 − W·M` read (2KV) and the per-chunk state hand-off (2K²V/C) that the old formula omitted. The matmul part
  now equals torch's FlopCounterMode on `ww_chunked` exactly for three shapes (new test
  `tests/test_model.py::test_ww_flop_formula_matches_flop_counter`). The triangular inverse stays at its intrinsic C²/3.
  Effect for `ww`: core 6,619 -> 6,708 FLOPs/token/head forward (+1.3%).
- Allowed by: PLAN.md §7 (bug fix + test).
- Affected runs: none.

## 2026-11-01 — WHAT×WHEN chunk size 32 -> 64 (same function, faster)
- What changed: `BASE_MODEL["ww"]["chunk"]` 64 (all ww arms). Mixer forward+backward at B16/T1024: chunk 16 358 ms,
  32 165 ms, **64 121 ms**, 128 147 ms (peak 4.0 / 4.0 / 5.2 / 8.6 GB). Tests pass (chunked == sequential).
  The analytic FLOPs for Q4 use the chunk actually run, so they now refer to C = 64.
- Why: A0.6 measured ww at 23,171 targets/s (chunk 32) vs tpp 607,200 (26× slower; PLAN §6 predicted ~4×).
  A profile shows device-to-device copies 17%, fp32 elementwise feature work ~35%, fp32 GEMMs ~32%, triangular
  solves ~3%; building the features directly in heads-first layout saved only 3%, so it was not adopted.
- Allowed by: PLAN.md §7 ("the WHAT×WHEN chunk size (it does not change the math)"); CLAUDE.md "When things go wrong".
- Affected runs: none.

## 2026-11-01 — A0.7 anchor outside 1.970 ± 0.05: explained, not changed
- What changed: nothing in the design. Three 8,192-update diagnostic runs of `tpp_legacy` outside the ledger (in the
  session scratchpad) to find the cause: old residual init (1.8263), old data order (2.0321), both (2.0231), against
  the anchor 1.8147. The old checkpoint re-scored by the new evaluator gives 1.9706 (old protocol 1.97019).
- Why: PLAN §4 A0.7 ("find the reason ... a documented, explained offset is acceptable"). Reason: the old project's
  training schedule draws each update's 64 sequences from a single book.
- Allowed by: PLAN.md §4 A0.7. ~18 min GPU.
- Affected runs: none.
