"""
run_figures_glm.py
===================
Orchestrate GLM group-comparison figures for one config.

Stages:
    A. Per-fish responder vs non-responder mean traces (tonic + phasic)
    B. Responder fraction boxplot   — 4-box [CTRL(+),EXPT(+),CTRL(-),EXPT(-)]
    C. ΔZ amplitude boxplot          — same layout, fixed-window effect size

B and C compute directly from f_tonic.npy + responder idx files (no cache
dependency) — lifted from the notebook's "Plot Box Plot comparing the
fraction of Tonic pos./neg responders" and "Quantify Amplitude Change" cells.

Usage
-----
    python run_figures_glm.py --config config_hcrt_trpv1_csn_120min

Location:
    ~/zwba/chemogenetic/run/run_figures_glm.py
"""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True,
                    help="Config module under chemogenetic/config/")
args, _ = parser.parse_known_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

COMPARISON_TAG        = cfg.COMPARISON_TAG
comparison_fig_dir    = cfg.comparison_fig_dir
baseline_end          = cfg.baseline_end
baseline_start        = cfg.baseline_start
CLIP_ABS_DZ           = cfg.CLIP_ABS_DZ
ctrl_fish             = cfg.ctrl_fish
CTRL_TAG              = cfg.CTRL_TAG
dir_analysis          = cfg.dir_analysis
drug_end              = cfg.drug_end
drug_start            = cfg.drug_start
expt_fish             = cfg.expt_fish
EXPT_TAG              = cfg.EXPT_TAG
NULL_TAG              = cfg.NULL_TAG
PLOT_META             = cfg.PLOT_META
RESPONDER_NULL_THRESH = 95 #cfg.RESPONDER_NULL_THRESH
sampling_rate_hz      = cfg.sampling_rate_hz

from utils.data_io import fish_dir

# ============================================================
# STAGE TOGGLES
# ============================================================
RUN_TRACES     = False   # disabled — F_phasic panel reused tonic responder idx,
                          # which conflates tonic and phasic classification.
                          # Re-enable once phasic/dprime responders are computed
                          # and cross-modal (tonic x phasic) categories are designed.
RUN_FRACTIONS  = True
RUN_DZ         = True

SHOW_PLOTS = False
SAVE_PLOTS = True

# ============================================================
# SHARED SETUP
# ============================================================
def frames_to_min_pair(start_frame, end_frame, sr_hz):
    return (float(start_frame) / sr_hz / 60.0, float(end_frame) / sr_hz / 60.0)

def minutes_to_frames(min_pair, sr_hz, Tfull):
    s = int(round(min_pair[0] * 60 * sr_hz))
    e = int(round(min_pair[1] * 60 * sr_hz))
    return int(np.clip(s, 0, Tfull)), int(np.clip(e, 0, Tfull))

# Apply 20-min offset to each epoch (skip habituation transient), matching dprime offset_sec=20*60.
OFFSET_MIN   = 20.0
_bl_start, _bl_end = frames_to_min_pair(baseline_start, baseline_end, sampling_rate_hz)
_dr_start, _dr_end = frames_to_min_pair(drug_start,     drug_end,     sampling_rate_hz)
BASELINE_MIN = (_bl_start + OFFSET_MIN, _bl_end)
DRUG_MIN     = (_dr_start + OFFSET_MIN, _dr_end)

ptag = int(RESPONDER_NULL_THRESH)
FIG_DIR = comparison_fig_dir(dir_analysis, COMPARISON_TAG)


def _load_responder_idx(fish):
    f_dir = fish_dir(dir_analysis, fish)
    pos_p = f_dir / f"tonic_pos_glm_{NULL_TAG}_nullp{ptag}_idxs.npy"
    neg_p = f_dir / f"tonic_neg_glm_{NULL_TAG}_nullp{ptag}_idxs.npy"
    if not pos_p.exists() or not neg_p.exists():
        return None, None
    return np.load(str(pos_p)), np.load(str(neg_p))


def p_to_stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"


def add_sig(ax, x1, x2, y, p):
    ax.plot([x1, x1, x2, x2], [y, y * 1.04, y * 1.04, y], lw=1.5, color="black")
    ax.text((x1 + x2) / 2, y * 1.06, f"{p_to_stars(p)} (p={p:.3g})",
             ha="center", va="bottom")


def _box_4group(data, labels, box_groups, ylabel, title, p_pos, p_neg, out_path):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhline(0, linestyle="--", linewidth=1.2, color="black", alpha=0.4, zorder=0)

    positions = [0, 1, 2, 3]
    bp = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        showfliers=False,
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
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    y_all = np.concatenate([d for d in data if len(d) > 0])
    ymax, ymin = float(np.nanmax(y_all)), float(np.nanmin(y_all))
    yrng = (ymax - ymin) if ymax > ymin else 1.0

    add_sig(ax, 0, 1, ymax + 0.10 * yrng, p_pos)
    add_sig(ax, 2, 3, ymax + 0.28 * yrng, p_neg)
    ax.margins(y=0.25)
    plt.tight_layout()

    if SAVE_PLOTS:
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        print(f"  ✅ Saved: {out_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ============================================================
# STAGE A: RESPONDER vs NON-RESPONDER MEAN TRACES
# ============================================================

BL_START_VOL = 1200   # vol 20 min — skip habituation transient

def _plot_responder_mean_traces(fish, group_tag):
    """
    Plot mean z-scored F_tonic and F_phasic traces split by GLM responder
    category: positive (red), negative (blue), non-responder (gray).
    Saves one PNG per fish.
    """
    proj_ID, expt_ID = fish
    base_dir = fish_dir(dir_analysis, fish)

    ft_path = base_dir / "f_tonic.npy"
    fp_path = base_dir / "f_phasic.npy"
    if not ft_path.exists() or not fp_path.exists():
        print(f"  ⚠️  {expt_ID}: f_tonic/f_phasic missing — skipping trace plot")
        return

    pos_idx, neg_idx = _load_responder_idx(fish)
    if pos_idx is None:
        print(f"  ⚠️  {expt_ID}: responder idx missing — skipping trace plot")
        return

    Ft = np.load(str(ft_path), mmap_mode="r").astype(np.float32)
    Fp = np.load(str(fp_path), mmap_mode="r").astype(np.float32)
    n_cells, T = Ft.shape
    t_min = np.arange(T) / (60 * sampling_rate_hz)

    resp_all = np.union1d(pos_idx, neg_idx)
    all_idx  = np.arange(n_cells)
    non_idx  = np.setdiff1d(all_idx, resp_all)

    def _zscore_group(arr, idx):
        if idx.size == 0:
            return None, None
        sub = arr[idx]
        mu  = sub[:, BL_START_VOL:int(baseline_end)].mean(axis=1, keepdims=True)
        sg  = sub[:, BL_START_VOL:int(baseline_end)].std(axis=1, keepdims=True).clip(1e-6)
        z   = (sub - mu) / sg
        z   = np.clip(z, -CLIP_ABS_DZ, CLIP_ABS_DZ)   # guard against near-zero σ outliers
        mean = z.mean(axis=0)
        sem  = z.std(axis=0) / np.sqrt(idx.size)
        return mean, sem

    groups = {
        f"Pos responders (N={pos_idx.size:,})": (pos_idx,  "#c0392b"),
        f"Neg responders (N={neg_idx.size:,})": (neg_idx,  "#2980b9"),
        f"Non-responders  (N={non_idx.size:,})": (non_idx, "#7f8c8d"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
    fig.suptitle(f"{expt_ID}  [{group_tag}] — Responder mean traces",
                 fontsize=11, fontweight="bold")

    for ax, arr, ylabel, title in zip(
        axes, [Ft, Fp],
        ["F_tonic (z-score)", "F_phasic (z-score)"],
        ["F_tonic", "F_phasic"],
    ):
        for label, (idx, color) in groups.items():
            mean, sem = _zscore_group(arr, idx)
            if mean is None:
                continue
            ax.plot(t_min, mean, color=color, lw=1.2, label=label, zorder=5)
            ax.fill_between(t_min, mean - sem, mean + sem,
                            color=color, alpha=0.2, zorder=4)

        ax.axvline(drug_start / (60 * sampling_rate_hz),
                   color="tomato",    lw=1.0, ls="--", alpha=0.8, label="drug start")
        ax.axvline(drug_end   / (60 * sampling_rate_hz),
                   color="steelblue", lw=1.0, ls="--", alpha=0.8, label="drug end")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(0, t_min[-1])
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    fish_fig_dir = base_dir / "figures"
    fish_fig_dir.mkdir(parents=True, exist_ok=True)
    out = fish_fig_dir / "responder_mean_traces.png"
    if SAVE_PLOTS:
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  ✅ {expt_ID} → {out.name}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    del Ft, Fp


# ============================================================
# STAGE B: RESPONDER FRACTION BOXPLOT
# ============================================================

def compute_group_fractions(fish_list, group_name):
    out_pos, out_neg = [], []
    for fish in fish_list:
        pos_idx, neg_idx = _load_responder_idx(fish)
        if pos_idx is None:
            print(f"  ⚠️  {fish[1]}: responder idx missing — skipping")
            continue
        f_dir = fish_dir(dir_analysis, fish)
        ft_path = f_dir / "f_tonic.npy"
        if not ft_path.exists():
            continue
        n_cells = np.load(str(ft_path), mmap_mode="r").shape[0]
        pos_frac = pos_idx.size / n_cells
        neg_frac = neg_idx.size / n_cells
        out_pos.append(pos_frac)
        out_neg.append(neg_frac)
        print(f"  {fish[1]} [{group_name}] n_cells={n_cells} "
              f"POS {pos_idx.size} ({pos_frac*100:.2f}%) "
              f"NEG {neg_idx.size} ({neg_frac*100:.2f}%)")
    return np.array(out_pos, dtype=np.float32), np.array(out_neg, dtype=np.float32)


def plot_responder_fraction(pos_ctrl, neg_ctrl, pos_expt, neg_expt):
    if any(len(x) == 0 for x in [pos_ctrl, neg_ctrl, pos_expt, neg_expt]):
        print("⚠️  Not enough data for responder fraction plot — skipping")
        return
    p_pos = mannwhitneyu(pos_expt, pos_ctrl, alternative="greater").pvalue
    p_neg = mannwhitneyu(neg_expt, neg_ctrl, alternative="greater").pvalue

    _box_4group(
        data=[pos_ctrl, pos_expt, neg_ctrl, neg_expt],
        labels=[
            f"{PLOT_META[CTRL_TAG]['label']} (+)",
            f"{PLOT_META[EXPT_TAG]['label']} (+)",
            f"{PLOT_META[CTRL_TAG]['label']} (−)",
            f"{PLOT_META[EXPT_TAG]['label']} (−)",
        ],
        box_groups=[CTRL_TAG, EXPT_TAG, CTRL_TAG, EXPT_TAG],
        ylabel="Fraction tonic responders per fish",
        title=f"Pos./Neg. Tonic GLM Responders Fraction "
              f"(ΔR² > pr{RESPONDER_NULL_THRESH} {NULL_TAG}-null)",
        p_pos=p_pos, p_neg=p_neg,
        out_path=FIG_DIR / f"glm_responder_fraction_{NULL_TAG}_p{ptag}.png",
    )
    print(f"  POS fraction MWU ({EXPT_TAG} > {CTRL_TAG}): p = {p_pos:.3e}")
    print(f"  NEG fraction MWU ({EXPT_TAG} > {CTRL_TAG}): p = {p_neg:.3e}")


# ============================================================
# STAGE C: ΔZ AMPLITUDE BOXPLOT (fixed-window effect size)
# ============================================================

def compute_group_dz(fish_list, group_name):
    out_pos, out_neg = [], []
    for fish in fish_list:
        pos_idx, neg_idx = _load_responder_idx(fish)
        if pos_idx is None:
            print(f"  ⚠️  {fish[1]}: responder idx missing — skipping")
            continue
        f_dir = fish_dir(dir_analysis, fish)
        ft_path = f_dir / "f_tonic.npy"
        if not ft_path.exists():
            continue

        Ft = np.load(str(ft_path), mmap_mode="r")
        n_cells, Tfull = Ft.shape
        b0, b1 = minutes_to_frames(BASELINE_MIN, sampling_rate_hz, Tfull)
        d0, d1 = minutes_to_frames(DRUG_MIN,     sampling_rate_hz, Tfull)

        pos_idx = pos_idx[(pos_idx >= 0) & (pos_idx < n_cells)]
        neg_idx = neg_idx[(neg_idx >= 0) & (neg_idx < n_cells)]

        def _dz_for(idxs):
            if idxs.size == 0:
                return np.nan
            sub = np.asarray(Ft[idxs, :], dtype=np.float32)
            mu_b = sub[:, b0:b1].mean(axis=1)
            sd_b = sub[:, b0:b1].std(axis=1)
            mu_d = sub[:, d0:d1].mean(axis=1)
            dz = (mu_d - mu_b) / (sd_b + 1e-6)
            dz = np.clip(dz, -CLIP_ABS_DZ, CLIP_ABS_DZ)
            dz = dz[np.isfinite(dz)]
            return float(dz.mean()) if dz.size else np.nan

        dz_pos = _dz_for(pos_idx)
        dz_neg = _dz_for(neg_idx)
        if np.isfinite(dz_pos):
            out_pos.append(dz_pos)
        if np.isfinite(dz_neg):
            out_neg.append(dz_neg)
        print(f"  {fish[1]} [{group_name}] ΔZ(+)={dz_pos:.3f}  ΔZ(−)={dz_neg:.3f}")
        del Ft

    return np.array(out_pos, dtype=np.float32), np.array(out_neg, dtype=np.float32)


def plot_dz_amplitude(dz_pos_ctrl, dz_neg_ctrl, dz_pos_expt, dz_neg_expt):
    if any(len(x) == 0 for x in [dz_pos_ctrl, dz_neg_ctrl, dz_pos_expt, dz_neg_expt]):
        print("⚠️  Not enough data for ΔZ amplitude plot — skipping")
        return
    p_pos = mannwhitneyu(dz_pos_expt, dz_pos_ctrl, alternative="greater").pvalue
    p_neg = mannwhitneyu(dz_neg_expt, dz_neg_ctrl, alternative="less").pvalue

    _box_4group(
        data=[dz_pos_ctrl, dz_pos_expt, dz_neg_ctrl, dz_neg_expt],
        labels=[
            f"{PLOT_META[CTRL_TAG]['label']} (+)",
            f"{PLOT_META[EXPT_TAG]['label']} (+)",
            f"{PLOT_META[CTRL_TAG]['label']} (−)",
            f"{PLOT_META[EXPT_TAG]['label']} (−)",
        ],
        box_groups=[CTRL_TAG, EXPT_TAG, CTRL_TAG, EXPT_TAG],
        ylabel="ΔZ (fixed-window effect size)",
        title=f"Tonic pos./neg. responder ΔZ | base {BASELINE_MIN[0]:.0f}-"
              f"{BASELINE_MIN[1]:.0f}min drug {DRUG_MIN[0]:.0f}-{DRUG_MIN[1]:.0f}min",
        p_pos=p_pos, p_neg=p_neg,
        out_path=FIG_DIR / f"glm_dz_amplitude_{NULL_TAG}_p{ptag}.png",
    )
    print(f"  ΔZ(+) MWU ({EXPT_TAG} > {CTRL_TAG}): p = {p_pos:.3e}")
    print(f"  ΔZ(−) MWU ({EXPT_TAG} < {CTRL_TAG}): p = {p_neg:.3e}")


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"run_figures_glm | null={NULL_TAG} p{RESPONDER_NULL_THRESH} | fig_dir={FIG_DIR}")

    if RUN_TRACES:
        print("\n── A: Responder vs non-responder mean traces ────────────")
        for fish in ctrl_fish:
            _plot_responder_mean_traces(fish, CTRL_TAG)
        for fish in expt_fish:
            _plot_responder_mean_traces(fish, EXPT_TAG)

    if RUN_FRACTIONS:
        print("\n── B: Responder fraction boxplot ────────────────────────")
        pos_ctrl, neg_ctrl = compute_group_fractions(ctrl_fish, CTRL_TAG)
        pos_expt, neg_expt = compute_group_fractions(expt_fish, EXPT_TAG)
        plot_responder_fraction(pos_ctrl, neg_ctrl, pos_expt, neg_expt)

    if RUN_DZ:
        print("\n── C: ΔZ amplitude boxplot ───────────────────────────────")
        dzp_ctrl, dzn_ctrl = compute_group_dz(ctrl_fish, CTRL_TAG)
        dzp_expt, dzn_expt = compute_group_dz(expt_fish, EXPT_TAG)
        plot_dz_amplitude(dzp_ctrl, dzn_ctrl, dzp_expt, dzn_expt)

    print(f"\n✅ run_figures_glm complete. Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
