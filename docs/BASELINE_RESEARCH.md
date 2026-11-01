# How the baselines were chosen (summary of the 2026-11-01 research report)

The full report ("Fair, strong baselines for a WHAT×WHEN wall-clock ladder on one RTX 3090") is in the owner's
claude.ai chat of 2026-11-01. This page keeps only what Stage A uses, and why. Results in the cited papers were
measured on A100/H100/TPU class hardware and subword data, not on an RTX 3090 at character level, so every transfer to
our setting is an assumption that Stage A partly tests.

## Decisions taken from it

| Decision in PLAN.md | Reason from the literature |
|---|---|
| Primary baseline = Transformer++ (RMSNorm, RoPE, SwiGLU, no biases, untied) | It is the control that linear-attention papers use (GLA, DeltaNet, Gated DeltaNet), so reviewers expect it |
| QK-norm in the baseline | Keeps small models stable over a wide LR range (Wortsman et al. 2023, models from 2.4M parameters) |
| Equal LR-sweep budget per arm, LR the only tuned knob | Under-tuning is the main threat at small scale (Porian et al. 2024); a fixed, equal search is the cheap fair version |
| WSD schedule, 20% (1 − sqrt) cooldown | Constant LR + cooldown matches cosine at every length; gains plateau near 20%; (1 − sqrt) beats linear (Hägele et al. 2024) |
| Gated DeltaNet (fla) as the recurrent reference | Delta rule + gating, beats Mamba-2 and DeltaNet in its paper, pip-installable Triton kernels (flash-linear-attention) |
| DeltaNet as a secondary reference | The closest published relative of WHAT×WHEN's β=1 delta update |
| Swap only the mixer, keep the backbone | The DeltaNet/GLA convention; isolates the addressing scheme; the old project showed FFNs matter |
| One hybrid arm with sliding-window attention | Hybrids close most of the recall gap of recurrent mixers (Zoology; Griffin; DeltaNet hybrids) |
| Sliding-window attention for long-context eval of the Transformer | Keeps RoPE within trained relative positions; the fair way to stream a 1024-trained Transformer |
| No FP8, no FlashAttention-3 | Not available on Ampere (SM86) |
| Muon only in Stage B | ~1.3–1.4× over tuned AdamW at 0.1B, ~1.1× at 1.2B, regime-dependent, no character-level evidence (Wen et al. 2025) |
| No data repetition in Stage A | Up to ~4 epochs is nearly free (Muennighoff et al. 2023), but Stage A never needs more than 0.2 epoch per run |

## Reference numbers (character/byte level, for orientation only)

- enwik8: 12-layer Transformer-XL, 41M params, 1.06 bpc; Al-Rfou et al. 12-layer, 44M, 1.11 bpc (Dai et al. 2019).
- PG-19 bytes, compute-controlled around 10^18–10^19 FLOPs: SpaceByte ~1.01, MegaByte ~1.08, byte Transformer ~1.14 bpb.
- Our ASCII-128 normalisation is not byte-identical to UTF-8 bpb, so these are not directly comparable to our BPC.

## Sources

- Hägele et al. 2024, Scaling laws and compute-optimal training beyond fixed training durations — https://arxiv.org/abs/2405.18392 (code: https://github.com/epfml/schedules-and-scaling)
- Wen et al. 2025, Fantastic pretraining optimizers and where to find them — https://arxiv.org/abs/2509.02046
- Wortsman et al. 2023, Small-scale proxies for large-scale Transformer training instabilities — https://arxiv.org/abs/2309.14322
- Porian et al. 2024, Resolving discrepancies in compute-optimal scaling of language models — https://mlanthology.org/neurips/2024/porian2024neurips-resolving/
- Muennighoff et al. 2023, Scaling data-constrained language models — https://arxiv.org/abs/2305.16264
- Yang et al. 2024, Gated Delta Networks — https://arxiv.org/abs/2412.06464 ; DeltaNet — https://arxiv.org/abs/2406.06484
- Yang et al. 2023, Gated Linear Attention — https://arxiv.org/abs/2312.06635
- Arora et al. 2023, Zoology — https://arxiv.org/abs/2312.04927
- De et al. 2024, Griffin — https://arxiv.org/abs/2402.19427
- flash-linear-attention — https://github.com/fla-org/flash-linear-attention
- modded-nanogpt (speedrun recipe, Stage B reference) — https://github.com/KellerJordan/modded-nanogpt
- Dai et al. 2019, Transformer-XL — https://arxiv.org/abs/1901.02860 ; Slagle 2024, SpaceByte — https://arxiv.org/abs/2404.14408
