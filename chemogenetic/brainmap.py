"""
brainmap.py
===========
Brain-map helpers and plotting for temporal intensity maps.

Helper functions (_coarse_shape, _voxelize_cell_values, _proj, _rotate,
_coarse_shape) are lifted verbatim from the notebook Step 7 / ALL CELLS cell.

plot_group_comparison() is the modular equivalent of the notebook's
final plotting block: 2-row-per-window (XY + YZ), 2-column (CTRL vs EXPT)
GridSpec layout, bwr colormap.

Location:
    ~/zwba/chemogenetic/brainmap.py
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ── settings (match notebook defaults) ────────────────────────────────────
DEFAULT_DS      = (5, 5, 2)    # coarse-grid downsampling (X, Y, Z)
DEFAULT_CMAP    = "bwr"
DEFAULT_VMIN    = -5.0
DEFAULT_VMAX    =  5.0
DEFAULT_PROJ    = "mean"        # projection mode for signed ΔZ
ROTATE_XY_MINUS_90 = True      # notebook default


# ── helpers — lifted verbatim from notebook ────────────────────────────────

def _coarse_shape(brain_shape, ds):
    """Compute coarse-grid shape after downsampling by ds."""
    brain_shape = np.array(brain_shape, dtype=int)
    ds = np.array(ds, dtype=int)
    return tuple(((brain_shape + ds - 1) // ds).astype(int))


def _voxelize_cell_values(coords, vals, brain_shape, ds):
    """
    Bin per-cell scalar values onto a coarse voxel grid.
    Lifted verbatim from notebook ALL CELLS cell.

    Parameters
    ----------
    coords : array (n_cells, 3)
        Template-space voxel coordinates [i, j, k].  NaN = invalid.
    vals : array (n_cells,)
        Per-cell scalar values (ΔZ or d').
    brain_shape : tuple (X, Y, Z)
        Full-resolution template brain shape.
    ds : tuple (dx, dy, dz)
        Downsampling factors per axis.

    Returns
    -------
    out : np.ndarray shape (Xc, Yc, Zc), dtype float32, or None if empty.
    """
    coords = np.asarray(coords)
    vals   = np.asarray(vals)

    mask   = np.isfinite(vals) & (~np.isnan(coords).any(axis=1))
    coords = coords[mask]
    vals   = vals[mask]

    X, Y, Z = brain_shape
    inb = (
        (coords[:, 0] >= 0) & (coords[:, 0] < X) &
        (coords[:, 1] >= 0) & (coords[:, 1] < Y) &
        (coords[:, 2] >= 0) & (coords[:, 2] < Z)
    )
    coords = coords[inb]
    vals   = vals[inb]

    if coords.shape[0] == 0:
        return None

    ds_arr    = np.array(ds, dtype=int)
    Xc, Yc, Zc = _coarse_shape((X, Y, Z), ds_arr)

    cidx    = np.floor(coords / ds_arr).astype(int)
    cidx[:, 0] = np.clip(cidx[:, 0], 0, Xc - 1)
    cidx[:, 1] = np.clip(cidx[:, 1], 0, Yc - 1)
    cidx[:, 2] = np.clip(cidx[:, 2], 0, Zc - 1)

    lin  = (cidx[:, 0] * (Yc * Zc) + cidx[:, 1] * Zc + cidx[:, 2]).astype(np.int64)
    nvox = Xc * Yc * Zc

    sumv = np.zeros(nvox, dtype=np.float64)
    cntv = np.zeros(nvox, dtype=np.int64)
    np.add.at(sumv, lin, vals.astype(np.float64))
    np.add.at(cntv, lin, 1)

    out    = np.full(nvox, np.nan, dtype=np.float32)
    m      = cntv > 0
    out[m] = (sumv[m] / cntv[m]).astype(np.float32)
    return out.reshape((Xc, Yc, Zc))


def _proj(vol, axis, mode="mean"):
    """Projection over one axis. Lifted verbatim from notebook."""
    if mode == "mean":
        return np.nanmean(vol, axis=axis)
    if mode == "max":
        return np.nanmax(vol, axis=axis)
    if mode == "maxabs":
        a    = np.abs(vol)
        a    = np.where(np.isfinite(a), a, -np.inf)
        idx  = np.nanargmax(a, axis=axis)
        vol0 = np.where(np.isfinite(vol), vol, np.nan)
        take = np.take_along_axis(vol0, np.expand_dims(idx, axis=axis), axis=axis)
        return np.squeeze(take, axis=axis)
    raise ValueError(f"Unknown proj mode: {mode!r}")


def _rotate(arr):
    """XY rotation applied before display. Lifted verbatim from notebook."""
    return np.rot90(arr, k=3) if ROTATE_XY_MINUS_90 else arr


# ── HCRT dot schematic positions ──────────────────────────────────────────
# 5 positions per side as (x_frac, y_frac) relative to (Xc, Yc) coarse grid.
# Placed at approximate anterior hypothalamus location in dorsal portrait view.

# ── main plotting function ─────────────────────────────────────────────────

def plot_group_comparison(
    ctrl_maps:       list,
    expt_maps:       list,
    window_labels:   list[str],
    brain_shape:     tuple,
    DS:              tuple  = DEFAULT_DS,
    ctrl_tag:        str    = "CTRL",
    expt_tag:        str    = "EXPT",
    diff_maps:       list | None = None,
    diff_tag:        str    = "EXPT − CTRL",
    metric_label:    str    = "Tonic ΔZ",
    title:           str    = "",
    cmap:            str    = DEFAULT_CMAP,
    vmin:            float  = DEFAULT_VMIN,
    vmax:            float  = DEFAULT_VMAX,
    proj_mode:       str    = DEFAULT_PROJ,
    # ── HCRT reference trace row ──────────────────────────────────────────
    hcrt_df                  = None,
    hcrt_trace_col:  str     = "hcrt_mean_z",
    hcrt_sem_col:    str     = "hcrt_sem_z",
    hcrt_fish_cols:  list    = None,
    hcrt_trace_label: str    = "Hcrt F_tonic",
    hcrt_window_vols: list   = None,
    hcrt_sampling_hz: float  = 1.0,
    # ─────────────────────────────────────────────────────────────────────
    fig_dir:         Path | None = None,
    filename:        str   = "temporal_intensity_map.png",
    show:            bool  = False,
) -> plt.Figure:
    """
    Temporal intensity map layout:
        Row 0 (opt): HCRT F_tonic or F_phasic trace spanning all columns
        Row 1+: CTRL | EXPT | EXPT-CTRL brain maps
        Cols: [XY portrait | YZ portrait-thin | GAP] per time window
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm

    n_windows  = len(window_labels)
    group_tags = [ctrl_tag, expt_tag]
    group_maps = [ctrl_maps, expt_maps]
    if diff_maps is not None:
        group_tags.append(diff_tag)
        group_maps.append(diff_maps)
    n_groups = len(group_tags)

    has_hcrt = hcrt_df is not None
    n_rows   = (1 if has_hcrt else 0) + n_groups

    Xc, Yc, Zc = _coarse_shape(brain_shape, DS)

    # columns: [XY, YZ, GAP] per window; last window has no GAP
    GAP_W      = max(1, int(Xc * 0.08))
    col_widths = []
    col_types  = []
    for wi in range(n_windows):
        col_widths.extend([Xc, Zc])
        col_types.extend(["xy", "yz"])
        if wi < n_windows - 1:
            col_widths.append(GAP_W)
            col_types.append("gap")
    n_cols = len(col_widths)

    xy_cols = [i for i, t in enumerate(col_types) if t == "xy"]
    yz_cols = [i for i, t in enumerate(col_types) if t == "yz"]

    # row heights
    hcrt_h   = Yc * 0.45
    brain_h  = Yc
    h_ratios = ([hcrt_h] if has_hcrt else []) + [brain_h] * n_groups

    scale = 2.0 / Xc
    fig_w = scale * sum(col_widths) + 1.6
    fig_h = scale * sum(h_ratios) + 1.0
    fig   = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        n_rows, n_cols,
        height_ratios = h_ratios,
        width_ratios  = col_widths,
        hspace        = 0.08,
        wspace        = 0.02,
        top           = 0.90,
        bottom        = 0.07,
        left          = 0.12,
        right         = 0.88,
    )

    norm    = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    last_im = None

    # ── HCRT trace row — single axis spanning all columns ─────────────────
    if has_hcrt:
        ax_hcrt = fig.add_subplot(gs[0, :])

        t_min  = hcrt_df["time_min"].values
        y_mean = hcrt_df[hcrt_trace_col].values
        y_sem  = hcrt_df[hcrt_sem_col].values \
                 if hcrt_sem_col in hcrt_df.columns else None
        fish_cols = hcrt_fish_cols or [
            c for c in hcrt_df.columns
            if c.startswith("fish") and c.endswith("_z")
        ]

        # 45–120 min absolute = 0–75 min relative to drug onset
        view_mask = (t_min >= 45) & (t_min <= 120)
        t_rel     = t_min[view_mask] - 45

        # shade temporal windows
        win_colors = plt.cm.tab10(np.linspace(0, 0.65, n_windows))
        if hcrt_window_vols is not None:
            for wi, (vs, ve) in enumerate(hcrt_window_vols):
                ts = vs / hcrt_sampling_hz / 60 - 45
                te = ve / hcrt_sampling_hz / 60 - 45
                ax_hcrt.axvspan(max(ts, 0), min(te, 75),
                                alpha=0.12, color=win_colors[wi], zorder=0)

        # washout start
        ax_hcrt.axvline(45, color="steelblue", lw=1.0, ls="--",
                        alpha=0.7, label="washout")

        # individual fish traces
        for fc in fish_cols:
            if fc in hcrt_df.columns:
                ax_hcrt.plot(t_rel, hcrt_df[fc].values[view_mask],
                             color="gray", lw=0.7, alpha=0.35)

        # mean ± SEM
        ax_hcrt.plot(t_rel, y_mean[view_mask],
                     color="black", lw=1.2, zorder=5)
        if y_sem is not None:
            ax_hcrt.fill_between(
                t_rel,
                (y_mean - y_sem)[view_mask],
                (y_mean + y_sem)[view_mask],
                color="black", alpha=0.15, zorder=4,
            )

        ax_hcrt.axhline(0, color="k", lw=0.5, ls=":")
        ax_hcrt.set_xlim(0, 75)
        ax_hcrt.set_xlabel("Time post drug onset (min)", fontsize=8)
        ax_hcrt.tick_params(labelsize=7)
        ax_hcrt.spines[["top", "right"]].set_visible(False)

        # label on left — close to plot left edge
        fig.text(gs[0, 0].get_position(fig).x0 - 0.01,
                 gs[0, 0].get_position(fig).y0
                 + gs[0, 0].get_position(fig).height / 2,
                 hcrt_trace_label, va="center", ha="right",
                 fontsize=9, fontweight="bold", rotation=90)

    # ── brain map rows ────────────────────────────────────────────────────
    brain_row_offset = 1 if has_hcrt else 0

    for gi, (gtag, gmaps) in enumerate(zip(group_tags, group_maps)):
        row = gi + brain_row_offset

        for wi in range(n_windows):
            vol   = gmaps[wi]
            ax_xy = fig.add_subplot(gs[row, xy_cols[wi]])
            ax_yz = fig.add_subplot(gs[row, yz_cols[wi]])

            if gi == 0:
                ax_xy.set_title(window_labels[wi], fontsize=9,
                                pad=3, fontweight="bold")

            if vol is None:
                ax_xy.axis("off")
                ax_yz.axis("off")
                continue

            # XY portrait: mean over Z → (Xc,Yc) → .T → (Yc,Xc)
            xy = _proj(vol, axis=2, mode=proj_mode)
            if np.isfinite(xy).any():
                last_im = ax_xy.imshow(
                    xy.T, cmap=cmap, norm=norm,
                    origin="upper", aspect="equal",
                    interpolation="nearest",
                )
            ax_xy.axis("off")

            # YZ portrait-thin: mean over X → (Yc,Zc)
            yz = _proj(vol, axis=0, mode=proj_mode)
            if np.isfinite(yz).any():
                last_im = ax_yz.imshow(
                    yz, cmap=cmap, norm=norm,
                    origin="upper", aspect="equal",
                    interpolation="nearest",
                )
            ax_yz.axis("off")

        # group label — close to plot left edge
        fig.text(
            gs[row, 0].get_position(fig).x0 - 0.01,
            gs[row, 0].get_position(fig).y0
            + gs[row, 0].get_position(fig).height / 2,
            gtag,
            va="center", ha="right",
            fontsize=9, fontweight="bold",
            multialignment="center",
        )

    # ── colorbar ─────────────────────────────────────────────────────────
    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.08, 0.016, 0.75])
        cbar    = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(metric_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)
    else:
        print("⚠️  No finite maps — colorbar skipped.")

    if title:
        fig.suptitle(title, fontsize=25, fontweight="bold")

    if fig_dir is not None:
        out = Path(fig_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  ✅ Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# APPEND the block below to ~/zwba/chemogenetic/brainmap.py
# ═══════════════════════════════════════════════════════════════════════════════


def _voxelize_presence(coords, responder_idx, brain_shape, ds):
    """
    Binary presence map per fish: 1 if ≥1 responder cell occupies this coarse
    voxel, else 0.  Uses OR/max logic — NOT averaging — because presence is a
    binary per-fish quantity.

    Summing these per-fish volumes across all fish in a group gives the
    N-fish count map used in the presence brain maps.

    Parameters
    ----------
    coords : np.ndarray (n_cells, 3), dtype int
        Template-space voxel coordinates.  -1 sentinel = invalid cell.
    responder_idx : np.ndarray (k,), dtype int
        Indices (into the n_cells axis) of cells that passed the IAAFT
        threshold (pos or neg).
    brain_shape : tuple (X, Y, Z)
        Full-resolution template brain shape, e.g. (280, 544, 40).
    ds : tuple (dx, dy, dz)
        Downsampling factors, e.g. (5, 5, 2).

    Returns
    -------
    out : np.ndarray (Xc, Yc, Zc), dtype float32
        Values in {0.0, 1.0}. Ready to be summed across fish to build the
        group count map.
    """
    cs = _coarse_shape(brain_shape, ds)
    out = np.zeros(cs, dtype=np.float32)

    if len(responder_idx) == 0:
        return out

    n_cells = coords.shape[0]

    # Build per-cell binary presence array.
    # Bounds-safe: clip responder_idx to valid range before indexing.
    presence = np.zeros(n_cells, dtype=np.float32)
    in_bounds = (responder_idx >= 0) & (responder_idx < n_cells)
    presence[responder_idx[in_bounds]] = 1.0

    # Drop cells with -1 sentinel coordinates.
    valid = (coords >= 0).all(axis=1)       # (n_cells,) bool
    coords_v   = coords[valid].astype(np.int32)   # (n_valid, 3)
    presence_v = presence[valid]                   # (n_valid,)

    xi = np.clip(coords_v[:, 0] // ds[0], 0, cs[0] - 1)
    yi = np.clip(coords_v[:, 1] // ds[1], 0, cs[1] - 1)
    zi = np.clip(coords_v[:, 2] // ds[2], 0, cs[2] - 1)

    # OR aggregation: any responder cell in a voxel → 1.
    np.maximum.at(out, (xi, yi, zi), presence_v)
    return out
