"""Figures for the gate report: `python -m ww.plots` -> results/figures/*.png

  curve_tokens.png   quick-validation BPC vs training targets (main arms, longest seed-0 run each)
  curve_time.png     the same against active training minutes, with the iso-time (A6) points
  lr_sweep.png       A1 sweep: final BPC vs learning rate
  ablations.png      A3: BPC difference to ww (seed 0); the shaded band is the +-0.03 noise zone
  longctx.png        per-position BPC at the longest evaluated length (A5)

Colours: fixed categorical order (tpp blue, gdn orange, ww aqua), validated for colour-vision
deficiency; every line is also labelled directly at its end, so identity never relies on colour.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ww.arms import ABLATIONS, MAIN_ARMS  # noqa: E402
from ww.gate import bpc, rows  # noqa: E402

from ww.paths import RESULTS, RUNS  # noqa: E402

FIG = os.path.join(RESULTS, "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"tpp": "#2a78d6", "gdn": "#eb6834", "ww": "#1baf7a"}
LABEL = {"tpp": "Transformer++", "gdn": "Gated DeltaNet", "ww": "WHAT×WHEN"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def new_fig(w=7.0, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(FIG, name))


def end_label(ax, x, y, text, color):
    ax.plot([x], [y], "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def longest_seed0_run(arm):
    for stage in ("A4_r40", "A2_r10"):
        for r in rows():
            if r["stage"] == stage and r["arm"] == arm and r["seed"] == 0 and r["status"] == "ok":
                path = os.path.join(RUNS, r["run"], "summary.json")
                if os.path.exists(path):
                    with open(path) as f:
                        return json.load(f)
    return None


def curves():
    runs = {a: longest_seed0_run(a) for a in MAIN_ARMS}
    if not any(runs.values()):
        return
    for xkey, fname, xlabel, scale in (("targets", "curve_tokens.png", "training targets (billions)", 1e9),
                                       ("active_seconds", "curve_time.png", "active training time (minutes)", 60.0)):
        fig, ax = new_fig()
        style(ax, "Validation BPC during training (seed 0, quick validation slice)", xlabel, "bits per character")
        for a, s in runs.items():
            if not s or not s["history"]:
                continue
            xs = [h[xkey] / scale for h in s["history"]]
            ys = [h["quick_val_bpc"] for h in s["history"]]
            ax.plot(xs, ys, color=COLOR[a], linewidth=2, label=LABEL[a])
            end_label(ax, xs[-1], ys[-1], f"{LABEL[a]} {ys[-1]:.3f}", COLOR[a])
        if xkey == "active_seconds":
            for a in MAIN_ARMS:
                for r in rows():
                    if r["stage"] == "A6_iso" and r["arm"] == a and r["status"] == "ok":
                        ax.plot([r["active_seconds"] / 60], [r["final_val_bpc"]], "D", color=COLOR[a],
                                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate("diamonds: separate 10 and 40 min runs (full validation)", (0.01, 0.02),
                        xycoords="axes fraction", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        save(fig, fname)


def lr_sweep():
    fig, ax = new_fig()
    style(ax, "Learning-rate sweep (A1: 268M targets, seed 0)", "peak learning rate", "final validation BPC")
    any_data = False
    for a in MAIN_ARMS:
        rs = sorted([r for r in rows() if r["stage"] == "A1_lr" and r["arm"] == a and r["status"] == "ok"],
                    key=lambda r: r["lr"])
        if not rs:
            continue
        any_data = True
        xs, ys = [r["lr"] for r in rs], [r["final_val_bpc"] for r in rs]
        ax.plot(xs, ys, "-o", color=COLOR[a], linewidth=2, markersize=5, label=LABEL[a])
        b = min(range(len(ys)), key=lambda i: ys[i])
        end_label(ax, xs[b], ys[b], f"best {xs[b]:.0e}", COLOR[a])
    if any_data:
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        save(fig, "lr_sweep.png")
    else:
        plt.close(fig)


def ablations():
    ref = bpc("A2_r10", "ww").get(0)
    items = [(a, bpc("A3_abl", a).get(0)) for a in ABLATIONS]
    items = [(a, b - ref) for a, b in items if b is not None and ref is not None]
    if not items:
        return
    items.sort(key=lambda t: t[1])
    fig, ax = new_fig(7.0, 0.45 * len(items) + 1.2)
    style(ax, "Ablations at R10 vs WHAT×WHEN (seed 0)", "BPC difference to ww (negative = better)", "")
    ax.axvspan(-0.03, 0.03, color=GRID, alpha=0.6, linewidth=0)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ys = range(len(items))
    ax.barh(list(ys), [d for _, d in items], height=0.6, color=COLOR["ww"])
    ax.set_yticks(list(ys))
    ax.set_yticklabels([a for a, _ in items], fontsize=8, color=INK)
    for y, (_, d) in zip(ys, items):
        ax.annotate(f"{d:+.3f}", (d, y), xytext=(4 if d >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if d >= 0 else "right", va="center", fontsize=8, color=INK)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    save(fig, "ablations.png")


def longctx():
    data = {}
    for a in MAIN_ARMS:
        s = longest_seed0_run(a)
        if s:
            p = os.path.join(RUNS, s["run"], "longctx.json")
            if os.path.exists(p):
                with open(p) as f:
                    data[a] = json.load(f)["curves"]
    if not data:
        return
    L = max(int(k) for d in data.values() for k in d)
    fig, ax = new_fig()
    style(ax, f"Loss by position in a {L}-character window (Transformer: strided 1024 windows)",
          "position in window (characters)", "BPC (log-binned)")
    for a, d in data.items():
        c = d.get(str(L))
        if not c:
            continue
        mids = [max(1, (x0 + x1) / 2) for x0, x1 in zip(c["edges"][:-1], c["edges"][1:])]
        ax.plot(mids, c["bpc"], color=COLOR[a], linewidth=2, label=LABEL[a])
        end_label(ax, mids[-1], c["bpc"][-1], LABEL[a], COLOR[a])
    ax.axvline(1024, color=INK2, linewidth=0.8, linestyle="--")
    ax.annotate("training length", (1024, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points",
                fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    save(fig, "longctx.png")


def main():
    curves()
    lr_sweep()
    ablations()
    longctx()


if __name__ == "__main__":
    main()
