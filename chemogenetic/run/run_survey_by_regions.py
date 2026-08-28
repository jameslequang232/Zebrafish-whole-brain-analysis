"""
run_survey_by_regions.py
========================
Regional anatomical survey of tonic ΔZ and phasic d' for two temporal
windows of interest:
    • Transition phase : 10–25 min post-capsaicin  (window index 1)
    • Stable phase     : 30–45 min post-capsaicin  (window index 3)

Pipeline
--------
Cell 1 — Ensure epoch_dz.npy / epoch_dprime.npy are cached for all fish
Cell 2 — Voxelize per fish (DS=(5,5,2), same as run_temporal_intensity_map.py),
          nanmean across fish → group-mean volume for EXPT and CTRL
Cell 3 — Compute EXPT − CTRL difference maps; split into pos / neg sub-volumes;
          save 20 volumes under survey_by_regions/:
              {metric}_{phase}_{EXPT/CTRL}_mean.npy   (8 files)
              {metric}_{phase}_diff_{mean/pos/neg}.npy (12 files)
Cell 4 — Load MapZebrain region masks (coarse grid, OR-pooled), rank regions
          by mean |EXPT − CTRL diff| inside each region mask; save 4 CSVs
Cell 5 — Print ranking tables
Cell 6 — Quadrant plots: top N regions by Euclidean (ΔZ, d') shift,
          CTRL (hollow) → EXPT (solid), saved to survey_by_regions/figures/
          (Adapted directly from notebook quadrant cell; uses full-res masks
           + cell-level coordinate lookup for per-region means)

Outputs under comparisons/<COMPARISON_TAG>/survey_by_regions/
--------------------------------------------------------------
    {metric}_{phase}_{EXPT/CTRL}_mean.npy      group-mean voxel volumes
    {metric}_{phase}_diff_{mean,pos,neg}.npy   EXPT − CTRL diff volumes
    tonic_{transition,stable}_ranking.csv       top regions by |diff|
    phasic_{transition,stable}_ranking.csv
    figures/quadrant_{transition,stable}_top{N}.png

Prerequisites
-------------
    run_decompose.py     →  f_tonic.npy, f_phasic.npy         (per fish)
    run_medoids.py       →  medoids_template_vox.npy           (per fish)
    run_temporal_intensity_map.py Cell 1
                         →  epoch_dz.npy, epoch_dprime.npy    (per fish)
    register_mapzebrain_to_template.py
                         →  <MASKS_DIR>/*_in_template.nii.gz  (region masks)

Usage
-----
    python chemogenetic/run/run_survey_by_regions.py \
        --config config_hcrt_trpv1_csn_120min

    python chemogenetic/run/run_survey_by_regions.py \
        --config config_hcrt_trpv1_csn_120min --overwrite --top_n 20

Location:
    ~/zwba/chemogenetic/run/run_survey_by_regions.py
"""

# %% ── imports ────────────────────────────────────────────────────────────

import argparse
import gc
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import ants
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# %% ── CLI ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Regional anatomical survey + quadrant plots for two temporal windows."
)
parser.add_argument("--config", default="config_hcrt_trpv1_csn_120min",
                    help="Config module under chemogenetic/config/")
parser.add_argument("--overwrite", action="store_true",
                    help="Recompute group volumes even if .npy files already exist.")
parser.add_argument("--top_n", type=int, default=15,
                    help="Top-N regions to show in the quadrant plot (default: 15).")
args, _ = parser.parse_known_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

from utils.data_io import fish_dir
from chemogenetic.temporal_windows import (
    TEMPORAL_WINDOWS,
    N_WINDOWS,
    load_epoch_dz,
    load_epoch_dprime,
    compute_epoch_dz,
    compute_epoch_dprime,
    load_ftonic,
    load_fphasic,
    save_epoch_dz,
    save_epoch_dprime,
)
from chemogenetic.brainmap import _voxelize_cell_values

# %% ── experiment config + shared constants ───────────────────────────────

dir_analysis = cfg.dir_analysis
EXPT_FISH    = cfg.expt_fish
CTRL_FISH    = cfg.ctrl_fish
EXPT_TAG     = cfg.EXPT_TAG
CTRL_TAG     = cfg.CTRL_TAG
CLIP_ABS_DZ  = cfg.CLIP_ABS_DZ

# MapZebrain masks registered into template space by register_mapzebrain_to_template.py
MASKS_DIR = Path(
    "/home/ychiu/yun/lightsheet/analysis_output/registration/"
    "300um_template/mapzebrain_in_template/"
)

OUT_DIR = (
    Path(dir_analysis) / "comparisons" / cfg.COMPARISON_TAG / "survey_by_regions"
)
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Voxelization — must match run_temporal_intensity_map.py
DS          = (5, 5, 2)
N_JOBS      = 25
BRAIN_SHAPE = (288, 568, 40)   # full-res template (X, Y, Z)

# Windows of interest
# TEMPORAL_WINDOWS: [(label, start_vol, end_vol), ...] — 7 entries
#   index 0 →  0-15 min
#   index 1 → 10-25 min  ← TRANSITION
#   index 2 → 20-35 min
#   index 3 → 30-45 min  ← STABLE
#   index 4 → 40-55 min
#   index 5 → 50-65 min
#   index 6 → 60-75 min
WINDOWS_OF_INTEREST = {
    "transition": 1,
    "stable":     3,
}
WINDOW_DISPLAY_LABELS = {
    "transition": "10–25 min (transition)",
    "stable":     "30–45 min (stable)",
}

TOP_N                = args.top_n
MIN_VOXELS_IN_COARSE = 3

# Plot styling — lifted from notebook
MARKER_SIZE = 90
DOT_ALPHA   = 0.8
LINE_ALPHA  = 0.6
LINE_WIDTH  = 1.5

all_groups = {EXPT_TAG: EXPT_FISH, CTRL_TAG: CTRL_FISH}

print("=" * 70)
print("run_survey_by_regions.py")
print(f"  Config   : {args.config}")
print(f"  Output   : {OUT_DIR}")
print(f"  Windows  : {WINDOWS_OF_INTEREST}")
print(f"  Top N    : {TOP_N}  (quadrant plot)")
print(f"  Ranking  : EXPT − CTRL diff, sorted by mean |diff| per region")
print("=" * 70)


# %% ── Cell 1: ensure epoch arrays exist for all fish ─────────────────────

print("\n── Cell 1: Ensuring epoch arrays are cached ──")
for group_tag, fish_list in all_groups.items():
    for fish in fish_list:
        f_dir = fish_dir(dir_analysis, fish)
        f_dir.mkdir(parents=True, exist_ok=True)
        if (f_dir / "epoch_dz.npy").exists() and (f_dir / "epoch_dprime.npy").exists():
            continue
        print(f"  Computing epochs for {fish[1]}...")
        f_tonic  = load_ftonic(f_dir)
        f_phasic = load_fphasic(f_dir)
        dz, _, _ = compute_epoch_dz(f_tonic)
        dprime   = compute_epoch_dprime(f_phasic)
        save_epoch_dz(dz, f_dir)
        save_epoch_dprime(dprime, f_dir)
        del f_tonic, f_phasic, dz, dprime
        gc.collect()
print("  ✅ All epoch arrays present.")


# %% ── Cell 2: voxelize per fish → group-mean volumes ─────────────────────

def _voxelize_fish_two_windows(fish, metric: str) -> dict:
    """Voxelize one fish at the two windows of interest; return {phase: coarse_vol}."""
    f_dir       = fish_dir(dir_analysis, fish)
    coords_path = f_dir / "medoids_template_vox.npy"
    result      = {phase: None for phase in WINDOWS_OF_INTEREST}

    if not coords_path.exists():
        print(f"  ⚠️  medoids_template_vox.npy missing for {fish[1]} — skipping")
        return result

    coords = np.load(str(coords_path))
    arr    = load_epoch_dz(f_dir) if metric == "dz" else load_epoch_dprime(f_dir)

    if arr.shape[0] != coords.shape[0]:
        print(f"  ⚠️  shape mismatch {fish[1]}: "
              f"arr={arr.shape[0]}, coords={coords.shape[0]} — skipping")
        return result

    for phase, w_idx in WINDOWS_OF_INTEREST.items():
        vals          = np.clip(arr[:, w_idx], -CLIP_ABS_DZ, CLIP_ABS_DZ)
        result[phase] = _voxelize_cell_values(coords, vals, BRAIN_SHAPE, DS)

    del coords, arr
    gc.collect()
    return result


def _build_group_mean(group_tag, fish_list, metric) -> dict:
    """Voxelize all fish in parallel, nanmean → {phase: float32_volume | None}."""
    print(f"  🚀 {group_tag} | {metric} ({len(fish_list)} fish)...")
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_voxelize_fish_two_windows)(fish, metric)
        for fish in tqdm(fish_list, desc=f"{group_tag} {metric}")
    )
    out = {}
    for phase in WINDOWS_OF_INTEREST:
        fish_vols = [r[phase] for r in results if r[phase] is not None]
        if not fish_vols:
            print(f"    ⚠️  No valid volumes for {group_tag} | {phase}")
            out[phase] = None
            continue
        out[phase] = np.nanmean(np.stack(fish_vols, axis=0), axis=0).astype(np.float32)
    return out


print("\n── Cell 2: Building group-mean volumes ──")

# all_means[metric][tag][phase] = float32 volume | None
all_means = {}

for metric in ("dz", "dprime"):
    all_means[metric] = {}
    for tag, fish_list in all_groups.items():
        probe = OUT_DIR / f"{metric}_transition_{tag}_mean.npy"
        if not args.overwrite and probe.exists():
            print(f"\n  ⏩ {tag} | {metric}: loading cached group means...")
            phases = {}
            for phase in WINDOWS_OF_INTEREST:
                p = OUT_DIR / f"{metric}_{phase}_{tag}_mean.npy"
                phases[phase] = np.load(p) if p.exists() else None
            all_means[metric][tag] = phases
        else:
            all_means[metric][tag] = _build_group_mean(tag, fish_list, metric)
            for phase, vol in all_means[metric][tag].items():
                if vol is not None:
                    p = OUT_DIR / f"{metric}_{phase}_{tag}_mean.npy"
                    np.save(p, vol)
                    print(f"    💾 {p.name}")

print("\n  ✅ Group means ready.")


# %% ── Cell 3: EXPT − CTRL diff + pos/neg splits → save ──────────────────

print("\n── Cell 3: Computing EXPT − CTRL difference maps ──")

# diff_vols[metric][phase] = {"diff": vol, "pos": vol, "neg": vol}
diff_vols = {}

for metric in ("dz", "dprime"):
    diff_vols[metric] = {}
    for phase in WINDOWS_OF_INTEREST:
        expt_vol = all_means[metric][EXPT_TAG][phase]
        ctrl_vol = all_means[metric][CTRL_TAG][phase]

        if expt_vol is None or ctrl_vol is None:
            print(f"  ⚠️  Missing volume for {metric} | {phase} — skipping diff")
            diff_vols[metric][phase] = {"diff": None, "pos": None, "neg": None}
            continue

        diff = (expt_vol - ctrl_vol).astype(np.float32)
        pos  = np.where(diff > 0, diff, 0.0).astype(np.float32)
        neg  = np.where(diff < 0, diff, 0.0).astype(np.float32)

        np.save(OUT_DIR / f"{metric}_{phase}_diff_mean.npy", diff)
        np.save(OUT_DIR / f"{metric}_{phase}_diff_pos.npy",  pos)
        np.save(OUT_DIR / f"{metric}_{phase}_diff_neg.npy",  neg)

        diff_vols[metric][phase] = {"diff": diff, "pos": pos, "neg": neg}
        print(f"  ✅ {metric} | {phase}: diff [{diff.min():.3f}, {diff.max():.3f}]"
              f" — saved diff_{{mean,pos,neg}}.npy")

print(f"\n  Files: 8 group-mean + 12 diff .npy → {OUT_DIR}")


# %% ── Cell 4: load region masks (coarse grid) + rank by mean |diff| ──────

def _coarse_shape(brain_shape, ds):
    return tuple(int(np.ceil(b / d)) for b, d in zip(brain_shape, ds))


def _load_region_masks_coarse() -> dict:
    """
    Load *_in_template.nii.gz masks, OR-pool into coarse DS grid.
    Returns {region_name: bool_array of shape _coarse_shape(BRAIN_SHAPE, DS)}.
    """
    mask_files = sorted(MASKS_DIR.glob("*_in_template.nii.gz"))
    if not mask_files:
        print(f"  ⚠️  No region masks found in {MASKS_DIR}")
        return {}

    print(f"\n── Cell 4a: Loading {len(mask_files)} masks → coarse grid DS={DS}")
    c_shape  = _coarse_shape(BRAIN_SHAPE, DS)
    regions  = {}

    for mf in tqdm(mask_files, desc="Downsampling masks"):
        name  = mf.stem.replace("_in_template", "")
        arr   = ants.image_read(str(mf), pixeltype="float").numpy()

        pad   = [(0, (-s % d) % d) for s, d in zip(arr.shape, DS)]
        arr_p = np.pad(arr, pad, mode="constant", constant_values=0)

        cX, cY, cZ = arr_p.shape[0]//DS[0], arr_p.shape[1]//DS[1], arr_p.shape[2]//DS[2]
        c_vol = (arr_p
                 .reshape(cX, DS[0], cY, DS[1], cZ, DS[2])
                 .max(axis=(1, 3, 5))
                 .astype(bool))
        c_vol = c_vol[:c_shape[0], :c_shape[1], :c_shape[2]]

        if c_vol.sum() >= MIN_VOXELS_IN_COARSE:
            regions[name] = c_vol

    print(f"  {len(regions)} regions with ≥{MIN_VOXELS_IN_COARSE} coarse voxels kept.")
    return regions


def rank_regions_by_diff(diff_vol, regions) -> pd.DataFrame:
    """Rank regions by mean |EXPT − CTRL diff| inside each region mask."""
    if diff_vol is None:
        return pd.DataFrame()
    rows = []
    for region_name, mask in regions.items():
        vals = diff_vol[mask]
        if vals.size == 0:
            continue
        pos_v = vals[vals > 0]
        neg_v = vals[vals < 0]
        rows.append({
            "region":        region_name,
            "mean_abs_diff": float(np.abs(vals).mean()),
            "mean_diff":     float(vals.mean()),
            "mean_pos_diff": float(pos_v.mean()) if pos_v.size else 0.0,
            "mean_neg_diff": float(neg_v.mean()) if neg_v.size else 0.0,
            "frac_pos":      float((vals > 0).mean()),
            "frac_neg":      float((vals < 0).mean()),
            "n_coarse_vox":  int(mask.sum()),
        })
    df = (pd.DataFrame(rows)
            .sort_values("mean_abs_diff", ascending=False)
            .reset_index(drop=True))
    df.index      = df.index + 1
    df.index.name = "rank"
    return df


regions_coarse = _load_region_masks_coarse()

RANKING_TARGETS = [
    ("dz",     "transition", "Tonic ΔZ",  "tonic_transition_ranking.csv"),
    ("dz",     "stable",     "Tonic ΔZ",  "tonic_stable_ranking.csv"),
    ("dprime", "transition", "Phasic d′", "phasic_transition_ranking.csv"),
    ("dprime", "stable",     "Phasic d′", "phasic_stable_ranking.csv"),
]

print("\n── Cell 4b: Ranking regions by mean |EXPT − CTRL| ──")
ranking_dfs = {}

for metric, phase, metric_label, csv_fname in RANKING_TARGETS:
    key      = f"{metric}_{phase}"
    diff_vol = diff_vols[metric][phase]["diff"]
    df       = rank_regions_by_diff(diff_vol, regions_coarse)
    if df.empty:
        print(f"  ⚠️  No ranking data for {key}")
        continue
    df.to_csv(OUT_DIR / csv_fname)
    ranking_dfs[key] = df
    print(f"  ✅ {metric_label} | {phase} → {csv_fname}  ({len(df)} regions)")


# %% ── Cell 5: print ranking tables ──────────────────────────────────────

TOP_N_PRINT = 20

print("\n" + "=" * 70)
print(f"REGION RANKING — top {TOP_N_PRINT} by mean |EXPT − CTRL diff|")
print("=" * 70)

DISPLAY = {
    "dz_transition":     ("Tonic ΔZ",  "10–25 min (transition)"),
    "dz_stable":         ("Tonic ΔZ",  "30–45 min (stable)"),
    "dprime_transition": ("Phasic d′", "10–25 min (transition)"),
    "dprime_stable":     ("Phasic d′", "30–45 min (stable)"),
}

for key, (metric_lbl, phase_lbl) in DISPLAY.items():
    df = ranking_dfs.get(key)
    if df is None or df.empty:
        print(f"\n── {metric_lbl} | {phase_lbl} : no data\n")
        continue
    top = df.head(TOP_N_PRINT)
    print(f"\n{'─'*70}")
    print(f"  {metric_lbl} | {phase_lbl}")
    print(f"{'─'*70}")
    print(f"  {'Rank':>4}  {'Region':<45}  {'|diff|':>7}  {'diff':>7}  "
          f"{'%pos':>6}  {'%neg':>6}  {'n_vox':>6}")
    print(f"  {'─'*4}  {'─'*45}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}")
    for rank, row in top.iterrows():
        print(f"  {rank:>4}  {row['region']:<45}  "
              f"{row['mean_abs_diff']:>7.4f}  {row['mean_diff']:>+7.4f}  "
              f"{row['frac_pos']*100:>5.1f}%  {row['frac_neg']*100:>5.1f}%  "
              f"{row['n_coarse_vox']:>6}")


# %% ── Cell 6: quadrant plots (adapted from notebook) ────────────────────
#
# Uses full-res masks + direct cell-level coordinate lookup (not coarse grid)
# to compute per-region (ΔZ, d') means per fish → group nanmean.
# Ranks regions by tonic or phasic CTRL→EXPT shift.

print("\n── Cell 6: Quadrant plots ──")

# ── helpers ───────────────────────────────────────────────────────────────

def _finite_1d(x):
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]


def _coords_to_int_vox(coords, brain_shape):
    c = np.asarray(coords)
    X, Y, Z = brain_shape
    ci = np.rint(c).astype(np.int32)
    ci[:, 0] = np.clip(ci[:, 0], 0, X - 1)
    ci[:, 1] = np.clip(ci[:, 1], 0, Y - 1)
    ci[:, 2] = np.clip(ci[:, 2], 0, Z - 1)
    return ci


# ── load masks at full resolution for coordinate lookup ───────────────────

def _load_region_masks_fullres() -> list:
    mask_files = sorted(MASKS_DIR.glob("*_in_template.nii.gz"))
    if not mask_files:
        raise FileNotFoundError(f"No region masks found in {MASKS_DIR}")
    print(f"  Loading {len(mask_files)} masks at full resolution...")
    out = []
    for mf in tqdm(mask_files, desc="Full-res masks", leave=False):
        name = mf.stem.replace("_in_template", "")
        arr  = ants.image_read(str(mf), pixeltype="float").numpy()
        if arr.shape != BRAIN_SHAPE:
            continue
        m = arr > 0
        if m.any():
            out.append((name, m))
    print(f"  {len(out)} non-empty full-res regions loaded.")
    return out


regions_fullres  = _load_region_masks_fullres()
region_names_all = [r[0] for r in regions_fullres]


# ── per-fish per-region means ──────────────────────────────────────────────

def _quadrant_fish(fish, w_idx: int) -> tuple:
    expt_id     = fish[1]
    f_dir       = fish_dir(dir_analysis, fish)
    coords_path = f_dir / "medoids_template_vox.npy"

    if not coords_path.exists():
        return expt_id, None

    coords = np.load(str(coords_path))
    dz_arr = load_epoch_dz(f_dir)
    dp_arr = load_epoch_dprime(f_dir)

    if coords.shape[0] != dz_arr.shape[0] or coords.shape[0] != dp_arr.shape[0]:
        return expt_id, None

    valid   = np.all(coords >= 0, axis=1)
    dz_vals = np.clip(dz_arr[:, w_idx], -CLIP_ABS_DZ, CLIP_ABS_DZ)
    dp_vals = dp_arr[:, w_idx]

    ci = _coords_to_int_vox(coords, BRAIN_SHAPE)
    xi, yi, zi = ci[:, 0], ci[:, 1], ci[:, 2]

    out = {}
    for rname, rmask in regions_fullres:
        in_reg = rmask[xi, yi, zi] & valid
        if in_reg.any():
            out[rname] = {
                "dz": float(np.nanmean(dz_vals[in_reg])),
                "dp": float(np.nanmean(dp_vals[in_reg])),
            }
        else:
            out[rname] = {"dz": np.nan, "dp": np.nan}

    del coords, dz_arr, dp_arr
    gc.collect()
    return expt_id, out


def _quadrant_group_summary(fish_list, group_tag, w_idx) -> dict:
    print(f"  🚀 {group_tag} ({len(fish_list)} fish)...")
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_quadrant_fish)(fish, w_idx)
        for fish in tqdm(fish_list, desc=group_tag, leave=False)
    )
    reg_dz = {r: [] for r in region_names_all}
    reg_dp = {r: [] for r in region_names_all}
    ok = 0
    for _, per_reg in results:
        if per_reg is None:
            continue
        ok += 1
        for rname in region_names_all:
            reg_dz[rname].append(per_reg[rname]["dz"])
            reg_dp[rname].append(per_reg[rname]["dp"])
    print(f"    ✅ {ok}/{len(fish_list)} fish valid.")
    summary = {}
    for rname in region_names_all:
        dz_v = _finite_1d(reg_dz[rname])
        dp_v = _finite_1d(reg_dp[rname])
        summary[rname] = {
            "dz_mean": float(np.nanmean(dz_v)) if dz_v.size else np.nan,
            "dp_mean": float(np.nanmean(dp_v)) if dp_v.size else np.nan,
        }
    return summary


def _rank_regions(ctrl_summary, expt_summary, top_n, rank_by) -> list:
    all_rows = []
    for rname in region_names_all:
        xc = ctrl_summary[rname]["dz_mean"]
        yc = ctrl_summary[rname]["dp_mean"]
        xe = expt_summary[rname]["dz_mean"]
        ye = expt_summary[rname]["dp_mean"]
        if all(np.isfinite(v) for v in (xc, yc, xe, ye)):
            all_rows.append((rname, xc, yc, xe, ye,
                             abs(xe - xc), abs(ye - yc)))
    sort_idx = 5 if rank_by == "tonic" else 6
    all_rows.sort(key=lambda r: r[sort_idx], reverse=True)
    return all_rows[:top_n]


def _quadrant_plot(top_rows, window_name, rank_by):
    """Build and save quadrant plot. Returns [(region_name, rgba), ...] for brain map."""
    window_label = WINDOW_DISPLAY_LABELS[window_name]
    rank_label   = ("tonic |ΔEXPT−CTRL ΔZ|" if rank_by == "tonic"
                    else "phasic |ΔEXPT−CTRL d′|")

    if not top_rows:
        print(f"  ⚠️  No regions with finite coords for {window_name}")
        return []

    # print table
    print(f"\n  Top {len(top_rows)} by {rank_label} | {window_label}")
    print(f"  {'#':>3}  {'Region':<40}  {'CTRL ΔZ':>8}  {'CTRL d′':>8}  "
          f"{'EXPT ΔZ':>8}  {'EXPT d′':>8}  {'rank_val':>9}")
    print(f"  {'─'*3}  {'─'*40}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")
    for i, row in enumerate(top_rows, 1):
        rname, xc, yc, xe, ye, ts, ps = row
        rv = ts if rank_by == "tonic" else ps
        print(f"  {i:>3}  {rname:<40}  {xc:>+8.4f}  {yc:>+8.4f}  "
              f"{xe:>+8.4f}  {ye:>+8.4f}  {rv:>9.4f}")

    cmap          = matplotlib.colormaps["gist_ncar"].resampled(max(len(top_rows), 1))
    region_colors = [(row[0], cmap(i)) for i, row in enumerate(top_rows)]

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    ax.axhline(0, color="black", lw=1.2)
    ax.axvline(0, color="black", lw=1.2)

    region_handles = []
    for i, (rname, xc, yc, xe, ye, _, __) in enumerate(top_rows):
        color = cmap(i)
        ax.plot([xc, xe], [yc, ye], color=color, lw=LINE_WIDTH,
                alpha=LINE_ALPHA, linestyle=":", zorder=2)
        ax.scatter(xc, yc, s=MARKER_SIZE, facecolors="none",
                   edgecolors=color, linewidths=2, alpha=DOT_ALPHA, zorder=3)
        ax.scatter(xe, ye, s=MARKER_SIZE, color=color,
                   edgecolors=color, alpha=DOT_ALPHA, zorder=4)
        region_handles.append(Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=color, markeredgecolor=color,
            markersize=7, label=rname,
        ))

    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none",
               markeredgecolor="black", markeredgewidth=2, markersize=8, label=CTRL_TAG),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="black",
               markeredgecolor="black", markersize=8, label=EXPT_TAG),
    ]
    ax.add_artist(ax.legend(handles=marker_handles, frameon=False,
                            loc="lower right", fontsize=9))
    ax.legend(handles=region_handles, title="Region", frameon=True,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=8, title_fontsize=9)

    ax.set_xlabel("Tonic ΔZ", fontsize=12)
    ax.set_ylabel("Phasic d′", fontsize=12)
    ax.set_title(
        f"Top {len(top_rows)} regions by {rank_label}\n"
        f"{EXPT_TAG} vs {CTRL_TAG} | {window_label}",
        fontsize=11,
    )
    all_x = _finite_1d([r[1] for r in top_rows] + [r[3] for r in top_rows])
    all_y = _finite_1d([r[2] for r in top_rows] + [r[4] for r in top_rows])
    if all_x.size and all_y.size:
        ax.set_xlim(-max(0.001, float(np.abs(all_x).max())) * 1.15,
                     max(0.001, float(np.abs(all_x).max())) * 1.15)
        ax.set_ylim(-max(0.2,   float(np.abs(all_y).max())) * 1.15,
                     max(0.2,   float(np.abs(all_y).max())) * 1.15)
    ax.set_box_aspect(1)

    out_path = FIG_DIR / f"quadrant_{window_name}_top{len(top_rows)}_by_{rank_by}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")

    return region_colors


# ── run quadrant for both windows × both ranking metrics ───────────────────

TOP5 = 10
panel_top5 = {}

for window_name, w_idx in WINDOWS_OF_INTEREST.items():
    print(f"\n  Window: {window_name}  ({WINDOW_DISPLAY_LABELS[window_name]})")

    print(f"  [CTRL — {CTRL_TAG}]")
    ctrl_s = _quadrant_group_summary(CTRL_FISH, CTRL_TAG, w_idx)

    print(f"  [EXPT — {EXPT_TAG}]")
    expt_s = _quadrant_group_summary(EXPT_FISH, EXPT_TAG, w_idx)

    for rank_by in ("tonic", "phasic"):
        top_rows      = _rank_regions(ctrl_s, expt_s, TOP_N, rank_by)
        region_colors = _quadrant_plot(top_rows, window_name, rank_by)
        panel_top5[(rank_by, window_name)] = region_colors[:TOP5]


# %% ── Cell 7: 4-panel EXPT−CTRL brain map with region outlines ───────────
#
# Layout:
#   top-left     tonic     transition    (ranked by tonic shift)
#   bottom-left  phasic    transition    (ranked by phasic shift)
#   top-right    tonic     stable
#   bottom-right phasic    stable
#
# Each panel:
#   • Diverging bwr colormap of mean-across-Z diff volume (EXPT − CTRL)
#   • Contour outlines of the top-5 region masks (Z-projected), color-matched
#     to the quadrant plot legend
#
# The diff volumes live on the coarse DS=(5,5,2) grid.
# Region masks are downsampled to the same coarse grid (already in
# regions_coarse from Cell 4) so contours align perfectly.

print("\n── Cell 7: 4-panel brain map with region outlines ──")

PANELS = [
    # (row, col, metric,   window,       rank_by,  title)
    (0, 0, "dz",     "transition", "tonic",  "Tonic ΔZ | 10\u201325 min"),
    (1, 0, "dprime", "transition", "phasic", "Phasic d\u2032 | 10\u201325 min"),
    (0, 1, "dz",     "stable",     "tonic",  "Tonic \u0394Z | 30\u201345 min"),
    (1, 1, "dprime", "stable",     "phasic", "Phasic d\u2032 | 30\u201345 min"),
]

# Separate color limits for tonic vs phasic
CLIM_PHASIC = 0.5   # fixed \u00b10.5 for phasic d'

# Tonic: shared 99th-pct across the two tonic panels
_tonic_abs = []
for _, _, _m, _w, _, _ in PANELS:
    if _m == "dz":
        _v = diff_vols[_m][_w]["diff"]
        if _v is not None:
            _tonic_abs.append(np.nanpercentile(np.abs(_v), 99))
CLIM_TONIC = float(np.nanmax(_tonic_abs)) if _tonic_abs else 1.0


def _z_project_coarse(vol3d):
    """Mean-project coarse (X,Y,Z) along Z; returns (Y,X) flipped so dorsal=top."""
    with np.errstate(all="ignore"):
        masked = np.where(vol3d == 0, np.nan, vol3d)
        proj   = np.nanmean(masked, axis=2)
    proj = np.nan_to_num(proj, nan=0.0)
    return proj.T   # (Y,X)


def _region_outline_proj(mask3d):
    """OR-project coarse bool mask along Z; returns (Y,X) with same flip."""
    return mask3d.any(axis=2).T[::-1, :]   # (Y,X) flipped to match origin="upper"


# Figure size: driven by actual coarse grid aspect ratio
_cs = _coarse_shape(BRAIN_SHAPE, DS)
_COARSE_X, _COARSE_Y = _cs[0], _cs[1]
_panel_w = 5.5
_panel_h = _panel_w * (_COARSE_Y / _COARSE_X)
fig_bm, axes = plt.subplots(2, 2,
                             figsize=(_panel_w * 2 + 1.5, _panel_h * 2 + 1.0),
                             layout="constrained")

for row, col, metric, window, rank_by, title in PANELS:
    ax       = axes[row, col]
    clim     = CLIM_PHASIC if metric == "dprime" else CLIM_TONIC
    diff_vol = diff_vols[metric][window]["diff"]

    if diff_vol is None:
        ax.set_visible(False)
        continue

    # brain map
    proj = _z_project_coarse(diff_vol)
    im   = ax.imshow(proj, cmap="bwr", vmin=-clim, vmax=clim,
                     origin="upper", aspect="equal",
                     interpolation="nearest")

    # region outlines
    top5 = panel_top5.get((rank_by, window), [])
    legend_handles = []
    for rname, color in top5:
        mask3d = regions_coarse.get(rname)
        if mask3d is None:
            continue
        outline = _region_outline_proj(mask3d).astype(float)
        if outline.any():
            ax.contour(outline, levels=[0.5], colors=[color],
                       linewidths=2, alpha=0.8)
        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=rname))

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="EXPT \u2212 CTRL")

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=6, framealpha=0.7,
                  loc="lower right", title="Top regions", title_fontsize=9)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("X (coarse vox)", fontsize=8)
    ax.set_ylabel("Y (coarse vox)", fontsize=8)
    ax.tick_params(labelsize=7)

fig_bm.suptitle(
    f"EXPT \u2212 CTRL intensity maps  |  {EXPT_TAG} vs {CTRL_TAG}\n"
    f"Region outlines: top {TOP5} by tonic shift (\u0394Z panels) "
    f"/ phasic shift (d\u2032 panels)",
    fontsize=16,
)

bm_path = FIG_DIR / f"brainmap_4panel_top{TOP5}_regions.png"
fig_bm.savefig(bm_path, dpi=300, bbox_inches="tight")
plt.close(fig_bm)
print(f"  \u2705 Saved: {bm_path}")



# %% ── summary ────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"Done. All outputs → {OUT_DIR}")
print(f"  Group means : {{metric}}_{{phase}}_{{EXPT/CTRL}}_mean.npy  (8 files)")
print(f"  Diff maps   : {{metric}}_{{phase}}_diff_{{mean,pos,neg}}.npy (12 files)")
print(f"  Rankings    : 4 × _ranking.csv")
print(f"  Quadrant    : figures/quadrant_{{transition,stable}}_top{TOP_N}_by_{{tonic,phasic}}.png  (4 files)")
print(f"  Brain map   : figures/brainmap_4panel_top{TOP5}_regions.png")
print("=" * 70)
