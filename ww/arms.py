"""Every arm of the study, defined once. `python -m ww.arms --write` regenerates configs/*.json.

All arms share size S: d_model 256, 4 layers, SwiGLU FFN (hidden 704), vocab 128, untied embeddings.
Only the sequence mixer changes, except where an ablation says otherwise. Non-embedding parameter
counts must match the Transformer++ arm within +-3% (checked by `python -m ww.arms --check`),
except the ablations marked `param_match: false`, whose difference is the point of the ablation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

BASE_MODEL = {
    "vocab": 128,
    "d_model": 256,
    "n_layers": 4,
    "layers": None,  # filled per arm
    "ffn": "swiglu",
    "ffn_hidden": 704,
    "tie_embeddings": False,
    "attn": {"n_heads": 4, "rope_base": 10000.0, "qk_norm": True, "window": None},
    "swa": {"window": 256},
    "ww": {"n_heads": 128, "value_dim": 1, "R": 2, "chunk": 64, "what": "learned", "when": "learned",
           "decay": False, "tick_param": "softplus"},
    "fla": {"n_heads": 4, "head_dim": 64, "expand_v": 1.0, "use_gate": True, "conv_size": 4},
}

BASE_TRAIN = {
    "optimizer": "adamw",
    "lr": None,  # chosen by the A1 sweep (results/lr_choice.json)
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "schedule": "wsd",  # linear warmup, constant, (1 - sqrt) cooldown to 0
    "warmup_frac": 0.02,
    "cooldown_frac": 0.2,
    "targets_per_update": 65536,
    "seq_len": 1024,
    "micro_batch": 16,  # A0 sets the largest that fits per arm; does not change the math
}


def _arm(layers, notes, lr_from=None, param_match=True, model=None, train=None, gpu_only=False):
    m = copy.deepcopy(BASE_MODEL)
    m["layers"] = layers
    m["n_layers"] = len(layers)
    for key, val in (model or {}).items():
        if isinstance(val, dict):
            m[key].update(val)
        else:
            m[key] = val
    t = copy.deepcopy(BASE_TRAIN)
    t.update(train or {})
    return {"model": m, "train": t, "notes": notes, "lr_from": lr_from, "param_match": param_match,
            "gpu_only": gpu_only}


A, W, G = "attn", "ww", "gdn"
ARMS = {
    # --- main arms -------------------------------------------------------------------------
    "tpp": _arm([A] * 4, "Transformer++ (primary baseline): RoPE, QK-norm, SwiGLU, untied, AdamW+WSD."),
    "gdn": _arm([G] * 4, "Gated DeltaNet (fla) in the same backbone: the mature delta-rule reference. "
                "FFN hidden 640 (not 704) to stay within +-3% params (A0.4).",
                gpu_only=True, model={"ffn_hidden": 640}),
    "ww": _arm([W] * 4, "WHAT×WHEN mixer in the same backbone (headline arm). H=128, V=1, beta=1."),
    # --- secondary reference -----------------------------------------------------------------
    "deltanet": _arm(["deltanet"] * 4, "DeltaNet (fla), ungated: WHAT×WHEN's closest published relative.",
                     lr_from="gdn", gpu_only=True),
    # --- WHAT×WHEN ablations (LR inherited from the tuned ww arm) ---------------------------
    "ww_nowhen": _arm([W] * 4, "No WHEN clock: content-only addressing (5 effective state dims/head).",
                      lr_from="ww", model={"ww": {"when": "none"}}),
    "ww_constclock": _arm([W] * 4, "Clock = position x learned per-head frequency (no token-driven ticks).",
                          lr_from="ww", model={"ww": {"when": "constant"}}),
    "ww_nowhat": _arm([W] * 4, "No WHAT angle: time-only addressing (5 effective state dims/head).",
                      lr_from="ww", model={"ww": {"what": "none"}}),
    "ww_noffn": _arm([W] * 4, "Banked design inside the harness: no FFN, H=384 heads to match params.",
                     lr_from="ww", model={"ffn": "none", "ww": {"n_heads": 384}}),
    "ww_wide": _arm([W] * 4, "More memory, less FFN: H=256 heads, FFN hidden 384 (params matched).",
                    lr_from="ww", model={"ffn_hidden": 384, "ww": {"n_heads": 256}}),
    "ww_decay": _arm([W] * 4, "Adds a learned forget gate alpha_t (never tested before). +1 control/head.",
                     lr_from="ww", param_match=False, model={"ww": {"decay": True}}),
    "ww_v8": _arm([W] * 4, "Vector payload V=8 with H=48 heads (tensor-core friendlier, larger state).",
                  lr_from="ww", model={"ww": {"n_heads": 48, "value_dim": 8}}),
    "ww_hybrid": _arm([W, W, "swa", W], "Hybrid: layer 3 of 4 is sliding-window attention (window 256).",
                      lr_from="ww"),
    # --- harness anchor (reproduces an old local number; not part of the comparison) ---------
    "tpp_legacy": _arm([A] * 2, "Old local recipe D256/L2 (budget_grid_v1 tr_d256_l2_swiglu_h672, seed 0: "
                       "1.9702 BPC at u8192): SwiGLU 672, no QK-norm, lr 3e-4, betas (0.9, 0.999), wd 0.01, "
                       "2% warmup + cosine. 1,623,296 params, as the old arm.",
                       param_match=False,
                       model={"ffn": "swiglu", "ffn_hidden": 672, "attn": {"qk_norm": False}},
                       train={"lr": 3e-4, "betas": [0.9, 0.999], "weight_decay": 0.01,
                              "schedule": "cosine"}),
}

MAIN_ARMS = ["tpp", "gdn", "ww"]
ABLATIONS = ["ww_nowhen", "ww_constclock", "ww_nowhat", "ww_noffn", "ww_wide", "ww_decay", "ww_v8",
             "ww_hybrid", "deltanet"]


def config_path(arm: str) -> str:
    return os.path.join(CONFIG_DIR, f"{arm}.json")


def load_arm(arm: str) -> dict:
    with open(config_path(arm)) as f:
        return json.load(f)


def write_configs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for name, cfg in ARMS.items():
        with open(config_path(name), "w") as f:
            json.dump({"arm": name, **cfg}, f, indent=2)
            f.write("\n")
    print(f"wrote {len(ARMS)} configs to {CONFIG_DIR}")


def check_params(arms=None, device="cpu"):
    """Print non-embedding parameter counts and flag arms outside +-3% of tpp."""
    import torch

    from ww.model import LM, param_report, state_bytes_per_seq
    ref = None
    rows = []
    for name in arms or ARMS:
        cfg = load_arm(name)
        if cfg.get("gpu_only") and device == "cpu":
            rows.append((name, None, None, None, "skipped on CPU (needs fla)"))
            continue
        model = LM(cfg["model"]).to(device)
        rep = param_report(model)
        if name == "tpp":
            ref = rep["params_nonemb"]
        state = state_bytes_per_seq(model, 1024)
        rows.append((name, rep["params_nonemb"], rep["params_total"], state, ""))
        del model
        if device != "cpu":
            torch.cuda.empty_cache()
    print(f"{'arm':16s} {'nonemb':>10s} {'total':>10s} {'vs tpp':>8s} {'state@1k':>10s}  note")
    bad = []
    for name, ne, tot, state, note in rows:
        if ne is None:
            print(f"{name:16s} {'-':>10s} {'-':>10s} {'-':>8s} {'-':>10s}  {note}")
            continue
        rel = (ne / ref - 1) if ref else 0.0
        must = load_arm(name).get("param_match", True)
        flag = "" if (not must or abs(rel) <= 0.03) else "  <-- OUTSIDE +-3%: adjust ffn_hidden"
        if flag:
            bad.append(name)
        print(f"{name:16s} {ne:10d} {tot:10d} {rel:+8.1%} {state / 1024:9.1f}K  {note}{flag}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="regenerate configs/*.json from ARMS")
    ap.add_argument("--check", action="store_true", help="print parameter counts and the +-3%% check")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.write:
        write_configs()
    if a.check:
        bad = check_params(device=a.device)
        raise SystemExit(1 if bad else 0)
