"""
run_glm_dprime_map.py
=====================
N-fish presence brain maps for tonic GLM and phasic d' responders.

Outputs (saved to {dir_analysis}/comparisons/{COMPARISON_TAG}/figures/):
    tonic_presence_map.png
    phasic_presence_map.png

Layout per PNG:
                     [POS]    [Reds CB]   [NEG]    [Blues CB]
                  XY     YZ     |       XY     YZ      |
    CTRL         ████   ██      |       ████   ██       |
    EXPT         ████   ██      |       ████   ██       |

Colorbars:
    Reds  (POS): centred in the gap column between POS and NEG groups.
    Blues (NEG): right margin, outside the GridSpec.
    Each row has its own independent colorbar (0 → n_fish for that group).

Usage
-----
    python run_glm_dprime_map.py --config config_hcrt_trpv1_csn_120min
    bash chemogenetic/submit/submit_glm_dprime_map.sh --config config_hcrt_trpv1_csn_120min

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_glm_dprime_map.py

PREREQUISITE: brainmap_presence_addition.py must be appended to chemogenetic/brainmap.py.
"""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True,
                    help="Config module name under chemogenetic/config/, "
                         "e.g. config_hcrt_trpv1_csn_120min")
args, _ = parser.parse_known_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

ctrl_fish      = cfg.ctrl_fish
expt_fish      = cfg.expt_fish
dir_analysis   = cfg.dir_analysis
CTRL_TAG       = cfg.CTRL_TAG
EXPT_TAG       = cfg.EXPT_TAG
COMPARISON_TAG = cfg.COMPARISON_TAG
NULL_TAG       = cfg.NULL_TAG
GLM_PTAG       = int(cfg.RESPONDER_NULL_THRESH)
DPRIME_PTAG    = int(getattr(cfg, "PHASIC_RESPONDER_NULL_THRESH", 99))

from chemogenetic.brainmap import _coarse_shape, _voxelize_presence   # noqa: E402

# ── pipeline parameters ───────────────────────────────────────────────────────
BRAIN_SHAPE = (280, 544, 40)   # (X, Y, Z) template — verify with ants.image_read().shape
DS          = (5, 5, 2)        # coarse downsampling (X, Y, Z)
DPI         = 150

DIR_ANALYSIS       = Path(dir_analysis)
COMPARISON_FIG_DIR = DIR_ANALYSIS / "comparisons" / COMPARISON_TAG / "figures"
COMPARISON_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── GridSpec layout constants (must stay in sync with plot_presence_map) ──────
_GS_TOP    = 0.92
_GS_BOT    = 0.05
_GS_HSPACE = 0.08
_GS_LEFT   = 0.11
_GS_RIGHT  = 0.82   # leaves right margin for Blues CB
_H_ROW = (_GS_TOP - _GS_BOT) / (2 + _GS_HSPACE)   # per-row height in fig fraction


# ── file-path helpers ─────────────────────────────────────────────────────────

def _fish_dir(proj_id, expt_id):
    return DIR_ANALYSIS / proj_id / expt_id


def _tonic_paths(proj_id, expt_id):
    d = _fish_dir(proj_id, expt_id)
    return (
        d / f"tonic_pos_glm_{NULL_TAG}_nullp{GLM_PTAG}_idxs.npy",
        d / f"tonic_neg_glm_{NULL_TAG}_nullp{GLM_PTAG}_idxs.npy",
    )


def _phasic_paths(proj_id, expt_id):
    d = _fish_dir(proj_id, expt_id)
    return (
        d / f"phasic_pos_dprime_iaaft_nullp{DPRIME_PTAG}_idxs.npy",
        d / f"phasic_neg_dprime_iaaft_nullp{DPRIME_PTAG}_idxs.npy",
    )


# ── load one fish ─────────────────────────────────────────────────────────────

def _load_fish(proj_id, expt_id, mode):
    coords_path = _fish_dir(proj_id, expt_id) / "medoids_template_vox.npy"
    pos_path, neg_path = (
        _tonic_paths(proj_id, expt_id) if mode == "tonic"
        else _phasic_paths(proj_id, expt_id)
    )

    missing = [p.name for p in [coords_path, pos_path, neg_path] if not p.exists()]
    if missing:
        print(f"    ⚠️  {expt_id}: missing {missing} — skipping")
        return None

    coords  = np.load(str(coords_path))
    pos_idx = np.load(str(pos_path)).astype(int)
    neg_idx = np.load(str(neg_path)).astype(int)
    print(f"    {expt_id}: {pos_idx.size:,} pos + {neg_idx.size:,} neg "
          f"/ {coords.shape[0]:,} cells")
    return coords, pos_idx, neg_idx


# ── group count maps ──────────────────────────────────────────────────────────

def compute_count_maps(fish_list, mode):
    cs        = _coarse_shape(BRAIN_SHAPE, DS)
    count_pos = np.zeros(cs, dtype=np.float32)
    count_neg = np.zeros(cs, dtype=np.float32)
    n_loaded  = 0

    for proj_id, expt_id in fish_list:
        result = _load_fish(proj_id, expt_id, mode)
        if result is None:
            continue
        coords, pos_idx, neg_idx = result
        count_pos += _voxelize_presence(coords, pos_idx, BRAIN_SHAPE, DS)
        count_neg += _voxelize_presence(coords, neg_idx, BRAIN_SHAPE, DS)
        n_loaded  += 1

    print(f"  → {n_loaded}/{len(fish_list)} fish loaded")
    return count_pos, count_neg, n_loaded


# ── projections ───────────────────────────────────────────────────────────────

def _make_projs(vol):
    """
    XY: max over Z → rot90(k=3) → (Yc, Xc)  [portrait: AP vertical, LR horizontal]
    YZ: max over X             → (Yc, Zc)  [strip: AP vertical, DV horizontal; no flip]
    Zero-count voxels → NaN for white background.
    """
    xy = np.rot90(np.max(vol, axis=2), k=3).astype(float)
    xy[xy == 0] = np.nan

    yz = np.max(vol, axis=0).astype(float)
    yz[yz == 0] = np.nan

    return xy, yz


# ── colorbar y-position helper ────────────────────────────────────────────────

def _cb_ypos(row_i, frac=0.72):
    """
    (y_bottom, height) in figure fraction for a CB centred on GridSpec row_i.
    row_i=0 → top row (CTRL); row_i=1 → bottom row (EXPT).
    """
    row_y0 = (_GS_TOP - _H_ROW) if row_i == 0 else _GS_BOT
    center = row_y0 + _H_ROW / 2
    h_cb   = _H_ROW * frac
    return center - h_cb / 2, h_cb


# ── figure ────────────────────────────────────────────────────────────────────

def plot_presence_map(
    ctrl_pos, ctrl_neg,
    expt_pos, expt_neg,
    n_ctrl, n_expt,
    metric_name,
    ptag,
    fig_dir,
    filename,
    dpi=150,
):
    """
    8-panel presence brain map.

    GridSpec (2 rows × 5 cols):
        col 0: XY_pos  col 1: YZ_pos  col 2: gap (Reds CB here)
        col 3: XY_neg  col 4: YZ_neg  [Blues CB in right margin]

    Colorbars are placed analytically from the GridSpec column positions:
        Reds CB  → centred in gap column (between POS and NEG groups)
        Blues CB → right margin outside GridSpec
    """
    # ── projections ───────────────────────────────────────────────────────
    c_xy_p, c_yz_p = _make_projs(ctrl_pos)
    c_xy_n, c_yz_n = _make_projs(ctrl_neg)
    e_xy_p, e_yz_p = _make_projs(expt_pos)
    e_xy_n, e_yz_n = _make_projs(expt_neg)

    Yc, Xc = c_xy_p.shape
    _,  Zc = c_yz_p.shape
    # Gap wide enough to comfortably hold the Reds colorbar
    gap = max(Zc, Xc // 3)

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.5, 9.5))
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        2, 5,
        figure=fig,
        width_ratios=[Xc, Zc, gap, Xc, Zc],
        height_ratios=[1, 1],
        left=_GS_LEFT, right=_GS_RIGHT,
        top=_GS_TOP,   bottom=_GS_BOT,
        hspace=_GS_HSPACE, wspace=0.04,
    )

    # ── axes ──────────────────────────────────────────────────────────────
    axs = {}
    for row_i in range(2):
        axs[(row_i, "xy_pos")] = fig.add_subplot(gs[row_i, 0])
        axs[(row_i, "yz_pos")] = fig.add_subplot(gs[row_i, 1])
        # col 2 = gap — no subplot (reserved for Reds CB)
        axs[(row_i, "xy_neg")] = fig.add_subplot(gs[row_i, 3])
        axs[(row_i, "yz_neg")] = fig.add_subplot(gs[row_i, 4])

    # ── compute colorbar x positions from GridSpec column layout ──────────
    total_w_units = Xc + Zc + gap + Xc + Zc
    avail_w  = _GS_RIGHT - _GS_LEFT
    unit_w   = avail_w / total_w_units

    # Reds CB: centred in the gap column (col 2)
    x_gap_l  = _GS_LEFT + (Xc + Zc) * unit_w          # left edge of gap col
    x_gap_r  = x_gap_l + gap * unit_w                  # right edge of gap col
    cb_w     = 0.018
    x_reds   = (x_gap_l + x_gap_r) / 2 - cb_w / 2     # centred in gap

    # Blues CB: right margin, outside GridSpec
    x_blues  = _GS_RIGHT + 0.03

    # ── imshow + per-row colorbars ─────────────────────────────────────────
    row_data = [
        (0, c_xy_p, c_yz_p, c_xy_n, c_yz_n, CTRL_TAG, n_ctrl),
        (1, e_xy_p, e_yz_p, e_xy_n, e_yz_n, EXPT_TAG, n_expt),
    ]

    for row_i, xy_p, yz_p, xy_n, yz_n, tag, n_fish in row_data:
        kw_p = dict(aspect="auto", interpolation="nearest", vmin=0, vmax=n_fish, cmap="Reds")
        kw_n = dict(aspect="auto", interpolation="nearest", vmin=0, vmax=n_fish, cmap="Blues")

        axs[(row_i, "xy_pos")].imshow(xy_p, **kw_p)
        axs[(row_i, "yz_pos")].imshow(yz_p, **kw_p)
        axs[(row_i, "xy_neg")].imshow(xy_n, **kw_n)
        axs[(row_i, "yz_neg")].imshow(yz_n, **kw_n)

        for role in ["xy_pos", "yz_pos", "xy_neg", "yz_neg"]:
            ax = axs[(row_i, role)]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_facecolor("white")

        # Row label
        pos_obj = axs[(row_i, "xy_pos")].get_position()
        mid_y   = (pos_obj.y0 + pos_obj.y1) / 2
        fig.text(0.01, mid_y, f"{tag}\n(N={n_fish})",
                 va="center", ha="left", fontsize=8, fontweight="bold", rotation=90)

        # Per-row colorbars at their respective x positions
        y_cb, h_cb = _cb_ypos(row_i)
        ticks = list(range(0, n_fish + 1))

        for x_cb, cmap_cb, lbl_cb in [
            (x_reds,  "Reds",  "# fish\n(POS)"),
            (x_blues, "Blues", "# fish\n(NEG)"),
        ]:
            cax = fig.add_axes([x_cb, y_cb, cb_w, h_cb])
            sm  = plt.cm.ScalarMappable(
                cmap=cmap_cb, norm=mcolors.Normalize(vmin=0, vmax=n_fish)
            )
            sm.set_array([])
            cb  = fig.colorbar(sm, cax=cax)
            cb.set_ticks(ticks)
            cb.ax.tick_params(labelsize=5)
            cb.set_label(lbl_cb, fontsize=6, labelpad=2)

    # ── XY / YZ sub-headers ───────────────────────────────────────────────
    for role, lbl in [("xy_pos", "XY"), ("yz_pos", "YZ"),
                       ("xy_neg", "XY"), ("yz_neg", "YZ")]:
        ax  = axs[(0, role)]
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.008, lbl,
                 ha="center", va="bottom", fontsize=7, color="#555555")

    # ── POS / NEG group headers ───────────────────────────────────────────
    for role_l, role_r, lbl, clr in [
        ("xy_pos", "yz_pos", f"▲ POS  ({metric_name})", "#cc2222"),
        ("xy_neg", "yz_neg", f"▼ NEG  ({metric_name})", "#1155cc"),
    ]:
        pos_l = axs[(0, role_l)].get_position()
        pos_r = axs[(0, role_r)].get_position()
        fig.text((pos_l.x0 + pos_r.x1) / 2, pos_l.y1 + 0.038, lbl,
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=clr)

    # ── suptitle ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"{metric_name} responders  |  N-fish presence map  |  "
        f"IAAFT p{ptag}  |  DS={DS}",
        fontsize=8.5, y=0.995, color="#222222",
    )

    out = Path(fig_dir) / filename
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → saved: {out.name}")


# ── delete old run_glm_brainmap outputs ───────────────────────────────────────

def _delete_old_outputs():
    old_patterns = [
        "glm_brainmap_dR2*.png", "glm_brainmap_sign*.png", "glm_brainmap_density*.png",
        "glm_dR2_map*.png",      "glm_sign_map*.png",      "glm_density_map*.png",
    ]
    deleted = []
    for pattern in old_patterns:
        for p in COMPARISON_FIG_DIR.glob(pattern):
            p.unlink()
            deleted.append(p.name)
    if deleted:
        print(f"  Deleted {len(deleted)} old output(s): {deleted}")
    else:
        print("  No old outputs found to delete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Config        : {args.config}")
    print(f"ctrl_fish     : {len(ctrl_fish)} fish")
    print(f"expt_fish     : {len(expt_fish)} fish")
    print(f"COMPARISON_TAG: {COMPARISON_TAG}")
    print(f"GLM  p{GLM_PTAG}  |  Phasic d′ p{DPRIME_PTAG}")
    print(f"Output → {COMPARISON_FIG_DIR}\n")

    print("Deleting old run_glm_brainmap outputs ...")
    _delete_old_outputs()

    for mode, metric_name, ptag, fname in [
        ("tonic",  "Tonic GLM", GLM_PTAG,    "tonic_presence_map.png"),
        ("phasic", "Phasic d′", DPRIME_PTAG, "phasic_presence_map.png"),
    ]:
        print(f"\n{'='*62}")
        print(f"  {metric_name} presence maps  (IAAFT p{ptag})")
        print(f"{'='*62}")

        print(f"\n── CTRL  ({len(ctrl_fish)} fish) ──")
        ctrl_pos, ctrl_neg, n_ctrl = compute_count_maps(ctrl_fish, mode)

        print(f"\n── EXPT  ({len(expt_fish)} fish) ──")
        expt_pos, expt_neg, n_expt = compute_count_maps(expt_fish, mode)

        print(f"\nPlotting → {fname} ...")
        plot_presence_map(
            ctrl_pos, ctrl_neg,
            expt_pos, expt_neg,
            n_ctrl=n_ctrl,
            n_expt=n_expt,
            metric_name=metric_name,
            ptag=ptag,
            fig_dir=COMPARISON_FIG_DIR,
            filename=fname,
            dpi=DPI,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
