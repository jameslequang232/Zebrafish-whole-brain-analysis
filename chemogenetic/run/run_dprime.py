"""
run_dprime.py
=============
SLURM entry point: compute phasic d′ per cell for all fish,
then save two group-level summary figures:

    Fig 1 — Phasic responder fractions
        Left panel  : fraction of cells with d′ ≥ +0.5 (positive responders)
        Right panel : fraction of cells with d′ ≤ -0.5 (negative responders)

    Fig 2 — Phasic d′ amplitude
        All cells / Top 5% (most +) / Top 1% (most +)
        mean d′ per fish, ctrl vs expt, Mann-Whitney U

Figures saved to:
    dir_analysis / comparisons / COMPARISON_TAG / figures /
        phasic_responder_fractions_{mode}.pdf
        phasic_dprime_amplitude_{mode}.pdf

Usage
-----
    sbatch submit_dprime.sh
    # or interactively:
    python run_dprime.py --config config_hcrt_trpv1_csn_120min

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_dprime.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True,
                    help="Config module under chemogenetic/config/")
args, _ = parser.parse_known_args()

import importlib
cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

all_fish          = cfg.all_fish
ctrl_fish         = cfg.ctrl_fish
expt_fish         = cfg.expt_fish
dir_analysis      = cfg.dir_analysis
baseline_start    = cfg.baseline_start
baseline_end      = cfg.baseline_end
drug_end          = cfg.drug_end
drug_start        = cfg.drug_start
INCLUDED_BASELINE = cfg.INCLUDED_BASELINE
sampling_rate_hz  = cfg.sampling_rate_hz
CTRL_TAG          = cfg.CTRL_TAG
EXPT_TAG          = cfg.EXPT_TAG
PLOT_META         = cfg.PLOT_META
COMPARISON_TAG    = cfg.COMPARISON_TAG

from chemogenetic.dprime import (
    dprime_one_fish,
    iaaft_null_dprime_one_fish,
    save_dprime_responder_idx,
    compute_ctrl_pooled_threshold,
)
from utils.data_io import fish_dir

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE      = False
AMPLITUDE_MODE = "raw"    # "raw" (signed, recommended), "abs", or "rms"
OFFSET_SEC     = 20 * 60  # skip first 20 min of each epoch window (habituation)
VAR_FLOOR      = 1e-4
EPS_VAR        = 1e-8
CLIP_ABS       = None

DRUG_END_EXTENDED = drug_end + 0 * 60   # no washout extension

N_JOBS = 28

# Responder threshold (fixed, used only for the summary figures below —
# NOT the IAAFT-null-based responder classification, which is separate)
DPRIME_POS_THRESH = +0.5
DPRIME_NEG_THRESH = -0.5

# ── IAAFT null stage toggles ────────────────────────────────
RUN_IAAFT_NULL   = False
RUN_RESPONDERS   = True
OVERWRITE_NULL       = False
OVERWRITE_RESPONDERS = True   # always overwrite so idx reflect current null

RESPONDER_NULL_THRESH = getattr(cfg, "PHASIC_RESPONDER_NULL_THRESH", 99)  # percentile, matches GLM convention
N_SURROGATES  = 200
N_ITER_IAAFT  = 50
N_CELLS_NULL  = 20000


# ============================================================
# WORKER
# ============================================================
def _run_one(fish):
    try:
        return dprime_one_fish(
            fish=fish,
            dir_analysis=dir_analysis,
            baseline_start=baseline_start,  # offset_sec handles 20-min skip
            baseline_end=baseline_end,
            drug_start=drug_start,
            drug_end=DRUG_END_EXTENDED,
            sampling_rate_hz=sampling_rate_hz,
            offset_sec=OFFSET_SEC,          # skip first 20 min of each epoch
            amplitude_mode=AMPLITUDE_MODE,
            var_floor=VAR_FLOOR,
            eps_var=EPS_VAR,
            clip_abs=CLIP_ABS,
            overwrite=OVERWRITE,
        )
    except Exception as e:
        return {"fish": fish, "status": f"ERROR: {e}"}


# ============================================================
# LOAD D′ ARRAYS PER FISH
# ============================================================
def _load_dprime(fish):
    """Load phasic_dprime_cells_{AMPLITUDE_MODE}.npy for one fish."""
    path = fish_dir(dir_analysis, fish) / f"phasic_dprime_cells_{AMPLITUDE_MODE}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing d′ file: {path}")
    return np.load(str(path))


# ============================================================
# STAT HELPER
# ============================================================
def _pval_label(p):
    if p < 0.001:
        return "*** (p<0.001)"
    elif p < 0.01:
        return f"** (p={p:.3f})"
    elif p < 0.05:
        return f"* (p={p:.3f})"
    else:
        return f"ns (p={p:.2f})"


def _mannwhitney(a, b):
    """Two-sided Mann-Whitney U; returns (stat, p)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(stat), float(p)


def _bracket(ax, x0, x1, y, label, color="black", fs=9):
    """Draw a significance bracket between two x positions."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=1.2, color=color)
    ax.text((x0 + x1) / 2, y + h * 1.2, label,
            ha="center", va="bottom", fontsize=fs, color=color)


# ============================================================
# FIG 1 — RESPONDER FRACTIONS
# ============================================================
def plot_responder_fractions(ctrl_dprime, expt_dprime, fig_dir, mode):
    """
    Two-panel boxplot: fraction of cells with d′ ≥ +0.5 (left)
    and d′ ≤ -0.5 (right), ctrl vs expt.

    Parameters
    ----------
    ctrl_dprime : list of np.ndarray, one per ctrl fish
    expt_dprime : list of np.ndarray, one per expt fish
    fig_dir     : Path
    mode        : str   (amplitude_mode tag for filename)
    """
    ctrl_meta = PLOT_META[CTRL_TAG]
    expt_meta = PLOT_META[EXPT_TAG]

    def _frac_pos(dp):
        finite = dp[np.isfinite(dp)]
        return np.mean(finite >= DPRIME_POS_THRESH) if finite.size > 0 else np.nan

    def _frac_neg(dp):
        finite = dp[np.isfinite(dp)]
        return np.mean(finite <= DPRIME_NEG_THRESH) if finite.size > 0 else np.nan

    ctrl_pos = np.array([_frac_pos(d) for d in ctrl_dprime])
    expt_pos = np.array([_frac_pos(d) for d in expt_dprime])
    ctrl_neg = np.array([_frac_neg(d) for d in ctrl_dprime])
    expt_neg = np.array([_frac_neg(d) for d in expt_dprime])

    fig, axes = plt.subplots(1, 2, figsize=(6, 4.5), sharey=False)

    for ax, ctrl_vals, expt_vals, title in zip(
        axes,
        [ctrl_pos, expt_pos],
        [ctrl_neg, expt_neg],
        [f"d′ ≥ {DPRIME_POS_THRESH}", f"d′ ≤ {DPRIME_NEG_THRESH}"],
    ):
        # swap: left panel = pos, right = neg
        pass  # handled below

    for ax, ctrl_vals, expt_vals, title in zip(
        axes,
        [ctrl_pos, ctrl_neg],
        [expt_pos, expt_neg],
        [f"d′ ≥ +{DPRIME_POS_THRESH}", f"d′ ≤ {DPRIME_NEG_THRESH}"],
    ):
        rng = np.random.default_rng(42)
        x_ctrl, x_expt = 0, 1

        # boxes
        bp = ax.boxplot(
            [ctrl_vals[np.isfinite(ctrl_vals)], expt_vals[np.isfinite(expt_vals)]],
            positions=[x_ctrl, x_expt],
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(marker=""),
        )
        bp["boxes"][0].set(facecolor=ctrl_meta["color"], alpha=ctrl_meta["alpha"])
        bp["boxes"][1].set(facecolor=expt_meta["color"], alpha=expt_meta["alpha"])

        # jittered scatter
        jitter_c = rng.uniform(-0.08, 0.08, size=len(ctrl_vals))
        jitter_e = rng.uniform(-0.08, 0.08, size=len(expt_vals))
        ax.scatter(x_ctrl + jitter_c, ctrl_vals,
                   color=ctrl_meta["color"], s=28, zorder=5, alpha=0.9,
                   edgecolors="black", linewidths=0.5)
        ax.scatter(x_expt + jitter_e, expt_vals,
                   color=expt_meta["color"], s=28, zorder=5, alpha=0.9,
                   edgecolors="black", linewidths=0.5)

        # stat
        _, p = _mannwhitney(ctrl_vals[np.isfinite(ctrl_vals)],
                            expt_vals[np.isfinite(expt_vals)])
        ymax = max(np.nanmax(ctrl_vals), np.nanmax(expt_vals))
        ax.set_ylim(bottom=0, top=ymax * 1.35)
        _bracket(ax, x_ctrl, x_expt, ymax * 1.12, _pval_label(p))

        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Fraction of cells / fish", fontsize=10)
        ax.set_xticks([x_ctrl, x_expt])
        ax.set_xticklabels([ctrl_meta["label"], expt_meta["label"]], fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Phasic Responder Fractions", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = fig_dir / f"phasic_responder_fractions_{mode}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out.name}")


# ============================================================
# FIG 2 — D′ AMPLITUDE (ALL / TOP 5% / TOP 1%)
# ============================================================
def plot_dprime_amplitude(ctrl_dprime, expt_dprime, fig_dir, mode):
    """
    Three-group boxplot: mean d′ per fish for all cells, top 5%, top 1%
    (ranked by most positive d′), ctrl vs expt.

    Parameters
    ----------
    ctrl_dprime : list of np.ndarray, one per ctrl fish
    expt_dprime : list of np.ndarray, one per expt fish
    fig_dir     : Path
    mode        : str
    """
    ctrl_meta = PLOT_META[CTRL_TAG]
    expt_meta = PLOT_META[EXPT_TAG]

    def _mean_top(dp, pct):
        """Mean d′ of the top `pct`% most positive cells."""
        finite = dp[np.isfinite(dp)]
        if finite.size == 0:
            return np.nan
        n = max(1, int(np.ceil(finite.size * pct / 100)))
        return float(np.mean(np.partition(finite, -n)[-n:]))

    def _mean_all(dp):
        finite = dp[np.isfinite(dp)]
        return float(np.nanmean(finite)) if finite.size > 0 else np.nan

    groups = [
        ("All cells",     [_mean_all(d)     for d in ctrl_dprime], [_mean_all(d)     for d in expt_dprime]),
        ("Top 5% (most +)", [_mean_top(d, 5)  for d in ctrl_dprime], [_mean_top(d, 5)  for d in expt_dprime]),
        ("Top 1% (most +)", [_mean_top(d, 1)  for d in ctrl_dprime], [_mean_top(d, 1)  for d in expt_dprime]),
    ]

    n_groups = len(groups)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    group_spacing = 2.2
    box_offset    = 0.5
    rng = np.random.default_rng(42)
    xtick_pos, xtick_lab = [], []

    all_vals = []
    for g_idx, (label, ctrl_vals, expt_vals) in enumerate(groups):
        ctrl_vals = np.array(ctrl_vals)
        expt_vals = np.array(expt_vals)
        xc = g_idx * group_spacing
        xe = xc + box_offset

        bp = ax.boxplot(
            [ctrl_vals[np.isfinite(ctrl_vals)], expt_vals[np.isfinite(expt_vals)]],
            positions=[xc, xe],
            widths=0.36,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(marker=""),
        )
        bp["boxes"][0].set(facecolor=ctrl_meta["color"], alpha=ctrl_meta["alpha"])
        bp["boxes"][1].set(facecolor=expt_meta["color"], alpha=expt_meta["alpha"])

        jc = rng.uniform(-0.07, 0.07, size=len(ctrl_vals))
        je = rng.uniform(-0.07, 0.07, size=len(expt_vals))
        ax.scatter(xc + jc, ctrl_vals, color=ctrl_meta["color"], s=28, zorder=5,
                   alpha=0.9, edgecolors="black", linewidths=0.5)
        ax.scatter(xe + je, expt_vals, color=expt_meta["color"], s=28, zorder=5,
                   alpha=0.9, edgecolors="black", linewidths=0.5)

        all_vals.extend(ctrl_vals[np.isfinite(ctrl_vals)].tolist())
        all_vals.extend(expt_vals[np.isfinite(expt_vals)].tolist())

        _, p = _mannwhitney(ctrl_vals[np.isfinite(ctrl_vals)],
                            expt_vals[np.isfinite(expt_vals)])

        # bracket y: will be set after full ylim is known; store for later
        groups[g_idx] = (label, ctrl_vals, expt_vals, xc, xe, p)

        xtick_pos += [xc, xe]
        xtick_lab += [ctrl_meta["label"], expt_meta["label"]]

    ax.axhline(0, color="gray", lw=0.8, ls="--")

    ymin = min(all_vals) if all_vals else -0.1
    ymax = max(all_vals) if all_vals else 1.0
    yrange = ymax - ymin
    ax.set_ylim(ymin - yrange * 0.08, ymax + yrange * 0.42)

    # draw brackets now that ylim is set
    bracket_y = ymax + yrange * 0.08
    for label, ctrl_vals, expt_vals, xc, xe, p in groups:
        _bracket(ax, xc, xe, bracket_y, _pval_label(p))

    # group labels above bracket
    label_y = bracket_y + yrange * 0.18
    for g_idx, (label, *_) in enumerate(groups):
        xc = g_idx * group_spacing
        xe = xc + box_offset
        ax.text((xc + xe) / 2, label_y, label,
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("d′", fontsize=11)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Phasic d′ Amplitude", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = fig_dir / f"phasic_dprime_amplitude_{mode}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out.name}")


# ============================================================
# FIG 3 — IAAFT-NULL RESPONDER FRACTIONS
# ============================================================

def _load_iaaft_responder_idx(fish, ptag):
    """Load (pos_idx, neg_idx, n_cells) using the IAAFT null classification."""
    f_dir = fish_dir(dir_analysis, fish)
    pos_p = f_dir / f"phasic_pos_dprime_iaaft_nullp{ptag}_idxs.npy"
    neg_p = f_dir / f"phasic_neg_dprime_iaaft_nullp{ptag}_idxs.npy"
    dp_p  = f_dir / f"phasic_dprime_cells_{AMPLITUDE_MODE}.npy"
    if not pos_p.exists() or not neg_p.exists() or not dp_p.exists():
        return None, None, None
    pos_idx = np.load(str(pos_p))
    neg_idx = np.load(str(neg_p))
    n_cells = np.load(str(dp_p), mmap_mode="r").shape[0]
    return pos_idx, neg_idx, n_cells


def plot_iaaft_responder_fractions(ctrl_fish_list, expt_fish_list, fig_dir, ptag, pooled_thr):
    """
    Two-panel boxplot: fraction of cells classified as pos/neg responders
    using a single ctrl-pooled IAAFT threshold applied to raw d′ for all fish.

    The pooled threshold is derived from ctrl fish null distributions only,
    so the question is "does this cell exceed ctrl-level noise?" for both groups.
    """
    ctrl_meta = PLOT_META[CTRL_TAG]
    expt_meta = PLOT_META[EXPT_TAG]

    def _fracs(fish_list):
        pos_f, neg_f = [], []
        for fish in fish_list:
            try:
                dp = _load_dprime(fish)
            except FileNotFoundError as e:
                print(f"  ⚠️  {e} — skipping")
                continue
            finite = dp[np.isfinite(dp)]
            if finite.size == 0:
                continue
            pos_f.append(float(np.mean(finite > pooled_thr)))
            neg_f.append(float(np.mean(finite < -pooled_thr)))
        return np.array(pos_f, dtype=float), np.array(neg_f, dtype=float)

    ctrl_pos, ctrl_neg = _fracs(ctrl_fish_list)
    expt_pos, expt_neg = _fracs(expt_fish_list)

    if any(len(x) == 0 for x in [ctrl_pos, ctrl_neg, expt_pos, expt_neg]):
        print("  ⚠️  Not enough data for IAAFT responder fraction plot — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6, 4.5), sharey=False)

    for ax, ctrl_vals, expt_vals, title in zip(
        axes,
        [ctrl_pos, ctrl_neg],
        [expt_pos, expt_neg],
        [f"d′ > +{pooled_thr:.3f} (ctrl-pooled p{ptag})",
         f"d′ < −{pooled_thr:.3f} (ctrl-pooled p{ptag})"],
    ):
        rng = np.random.default_rng(42)
        x_ctrl, x_expt = 0, 1

        bp = ax.boxplot(
            [ctrl_vals[np.isfinite(ctrl_vals)], expt_vals[np.isfinite(expt_vals)]],
            positions=[x_ctrl, x_expt], widths=0.45, patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="black"), capprops=dict(color="black"),
            flierprops=dict(marker=""),
        )
        bp["boxes"][0].set(facecolor=ctrl_meta["color"], alpha=ctrl_meta["alpha"])
        bp["boxes"][1].set(facecolor=expt_meta["color"], alpha=expt_meta["alpha"])

        jitter_c = rng.uniform(-0.08, 0.08, size=len(ctrl_vals))
        jitter_e = rng.uniform(-0.08, 0.08, size=len(expt_vals))
        ax.scatter(x_ctrl + jitter_c, ctrl_vals, color=ctrl_meta["color"], s=28,
                   zorder=5, alpha=0.9, edgecolors="black", linewidths=0.5)
        ax.scatter(x_expt + jitter_e, expt_vals, color=expt_meta["color"], s=28,
                   zorder=5, alpha=0.9, edgecolors="black", linewidths=0.5)

        _, p = _mannwhitney(ctrl_vals[np.isfinite(ctrl_vals)],
                            expt_vals[np.isfinite(expt_vals)])
        ymax = max(np.nanmax(ctrl_vals), np.nanmax(expt_vals))
        ax.set_ylim(bottom=0, top=ymax * 1.35)
        _bracket(ax, x_ctrl, x_expt, ymax * 1.12, _pval_label(p))

        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Fraction of cells / fish", fontsize=10)
        ax.set_xticks([x_ctrl, x_expt])
        ax.set_xticklabels([ctrl_meta["label"], expt_meta["label"]], fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Phasic Responder Fractions (ctrl-pooled IAAFT p{ptag})",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = fig_dir / f"phasic_responder_fractions_iaaft_p{ptag}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out.name}")


# ============================================================
# FIG 4 — IAAFT-NULL RESPONDER AMPLITUDE (mean d′ within responders)
# ============================================================

def plot_iaaft_responder_amplitude(ctrl_fish_list, expt_fish_list, fig_dir, ptag, pooled_thr):
    """
    4-box plot: mean d′ within pos/neg responder cells classified by the
    ctrl-pooled IAAFT threshold, ctrl vs expt.
    """
    ctrl_meta = PLOT_META[CTRL_TAG]
    expt_meta = PLOT_META[EXPT_TAG]

    def _amps(fish_list):
        pos_amp, neg_amp = [], []
        for fish in fish_list:
            try:
                dp = _load_dprime(fish)
            except FileNotFoundError as e:
                print(f"  ⚠️  {e} — skipping")
                continue
            finite = dp[np.isfinite(dp)]
            if finite.size == 0:
                continue
            pos_cells = finite[finite > pooled_thr]
            neg_cells = finite[finite < -pooled_thr]
            if pos_cells.size > 0:
                pos_amp.append(float(np.mean(pos_cells)))
            if neg_cells.size > 0:
                neg_amp.append(float(np.mean(neg_cells)))
        return np.array(pos_amp, dtype=float), np.array(neg_amp, dtype=float)

    ctrl_pos, ctrl_neg = _amps(ctrl_fish_list)
    expt_pos, expt_neg = _amps(expt_fish_list)

    if any(len(x) == 0 for x in [ctrl_pos, ctrl_neg, expt_pos, expt_neg]):
        print("  ⚠️  Not enough data for IAAFT amplitude plot — skipping")
        return

    _, p_pos = _mannwhitney(ctrl_pos, expt_pos)
    _, p_neg = _mannwhitney(ctrl_neg, expt_neg)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhline(0, linestyle="--", linewidth=1.2, color="black", alpha=0.4, zorder=0)

    data = [ctrl_pos, expt_pos, ctrl_neg, expt_neg]
    labels = [f"{ctrl_meta['label']} (+)", f"{expt_meta['label']} (+)",
              f"{ctrl_meta['label']} (−)", f"{expt_meta['label']} (−)"]
    positions = [0, 1, 2, 3]
    box_groups = [CTRL_TAG, EXPT_TAG, CTRL_TAG, EXPT_TAG]

    bp = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )
    for patch, gname in zip(bp["boxes"], box_groups):
        meta = PLOT_META[gname]
        patch.set_facecolor(meta["color"])
        patch.set_alpha(meta["alpha"])
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(0)
    for x0, vals, gname in zip(positions, data, box_groups):
        meta = PLOT_META[gname]
        x = x0 + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x, vals, s=38, color=meta["color"], alpha=max(meta["alpha"], 0.85),
                   edgecolor="black", linewidth=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean d′ within responder cells")
    ax.set_title(f"Phasic responder d′ amplitude (ctrl-pooled IAAFT p{ptag}, thr=±{pooled_thr:.3f})")

    y_all = np.concatenate([d for d in data if len(d) > 0])
    ymax, ymin = float(np.nanmax(y_all)), float(np.nanmin(y_all))
    yrng = (ymax - ymin) if ymax > ymin else 1.0
    _bracket(ax, 0, 1, ymax + 0.10 * yrng, _pval_label(p_pos))
    _bracket(ax, 2, 3, ymax + 0.28 * yrng, _pval_label(p_neg))
    ax.margins(y=0.25)
    plt.tight_layout()

    out = fig_dir / f"phasic_dprime_amplitude_iaaft_p{ptag}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out.name}")


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Running d′ for {len(all_fish)} fish (n_jobs={N_JOBS})")
    print(f"  dir_analysis   : {dir_analysis}")
    print(f"  amplitude_mode : {AMPLITUDE_MODE}")
    print(f"  offset_sec     : {OFFSET_SEC // 60} min")
    print(f"  baseline       : frames [{baseline_start}, {baseline_end})")
    print(f"  drug           : frames [{drug_start}, {DRUG_END_EXTENDED})")
    print()

    # ── Compute d′ ───────────────────────────────────────────
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_run_one)(fish) for fish in tqdm(all_fish, desc="d′ (fish)")
    )

    print("\n── Summary ──────────────────────────────")
    for r in results:
        status = r["status"]
        if status == "ok":
            print(
                f"  {r['fish'][1]:50s}  "
                f"mean={r['mean']:+.3f}  median={r['median']:+.3f}  "
                f"n={r['n_cells']}"
            )
        else:
            print(f"  {r['fish'][1]:50s}  {status}")

    # ── IAAFT null (window-label permutation, ctrl fish only) ───
    # Only ctrl fish need per-fish null distributions; the pooled ctrl
    # threshold derived from these is then applied to all fish for figures.
    if RUN_IAAFT_NULL:
        print("\n── d′ IAAFT null (ctrl fish only) ────────")
        for fish in ctrl_fish:
            try:
                r = iaaft_null_dprime_one_fish(
                    fish=fish,
                    dir_analysis=dir_analysis,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                    drug_start=drug_start,
                    drug_end=DRUG_END_EXTENDED,
                    sampling_rate_hz=sampling_rate_hz,
                    offset_sec=OFFSET_SEC,
                    amplitude_mode=AMPLITUDE_MODE,
                    var_floor=VAR_FLOOR,
                    eps_var=EPS_VAR,
                    n_surrogates=N_SURROGATES,
                    n_iter_iaaft=N_ITER_IAAFT,
                    n_cells_null=N_CELLS_NULL,
                    null_percentile=RESPONDER_NULL_THRESH,
                    overwrite=OVERWRITE_NULL,
                )
                print(f"  {fish[1]:50s}  {r['status']}  thr=±{r['thresh']:.4f}")
            except Exception as e:
                print(f"  {fish[1]:50s}  ERROR: {e}")

    # ── Ctrl-pooled IAAFT threshold ───────────────────────────
    # Computed here so it's available for both responder idx and figures.
    ptag = int(RESPONDER_NULL_THRESH)
    print("\n── Computing ctrl-pooled IAAFT threshold ────")
    try:
        pooled_thr = compute_ctrl_pooled_threshold(
            ctrl_fish, dir_analysis, null_percentile=RESPONDER_NULL_THRESH
        )
    except RuntimeError as e:
        print(f"  ❌ {e} — skipping responder indices and IAAFT figures.")
        pooled_thr = None

    # ── Responder indices (all fish, ctrl-pooled threshold) ───
    # All fish get responder indices saved using the ctrl-pooled threshold
    # so downstream brain mapping uses a consistent classification.
    # Ctrl fish additionally have their per-fish IAAFT null files on disk
    # for QC purposes, but the saved idx here use the pooled threshold.
    if RUN_RESPONDERS and pooled_thr is not None:
        print("\n── d′ responder indices (all fish, ctrl-pooled thr) ─")

        def _responders(fish):
            try:
                return save_dprime_responder_idx(
                    fish=fish,
                    dir_analysis=dir_analysis,
                    amplitude_mode=AMPLITUDE_MODE,
                    null_percentile=RESPONDER_NULL_THRESH,
                    external_thr=pooled_thr,
                    overwrite=OVERWRITE_RESPONDERS,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        resp_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_responders)(fish) for fish in tqdm(all_fish, desc="d′ responders")
        )
        for r in resp_results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    # ── Load d′ arrays ───────────────────────────────────────
    print("\n── Loading d′ arrays for figures ────────")
    ctrl_dprime, expt_dprime = [], []

    for fish in ctrl_fish:
        try:
            ctrl_dprime.append(_load_dprime(fish))
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")

    for fish in expt_fish:
        try:
            expt_dprime.append(_load_dprime(fish))
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")

    if not ctrl_dprime or not expt_dprime:
        print("  ❌ Insufficient data for figures — aborting plot stage.")
        return

    # ── Figure output dir ────────────────────────────────────
    fig_dir = Path(dir_analysis) / "comparisons" / COMPARISON_TAG / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n── Saving figures → {fig_dir}")

    # ── Fig 1: responder fractions ───────────────────────────
    plot_responder_fractions(ctrl_dprime, expt_dprime, fig_dir, AMPLITUDE_MODE)

    # ── Fig 2: d′ amplitude (fixed ±0.5 threshold) ───────────
    plot_dprime_amplitude(ctrl_dprime, expt_dprime, fig_dir, AMPLITUDE_MODE)

    # ── Fig 3: responder fractions (ctrl-pooled IAAFT null) ──
    if pooled_thr is not None:
        plot_iaaft_responder_fractions(ctrl_fish, expt_fish, fig_dir, ptag, pooled_thr)

        # ── Fig 4: responder amplitude (ctrl-pooled IAAFT null) ──
        plot_iaaft_responder_amplitude(ctrl_fish, expt_fish, fig_dir, ptag, pooled_thr)

    print("\nDone.")


if __name__ == "__main__":
    main()
