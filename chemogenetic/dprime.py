"""
dprime.py
=========
Phasic d′ (sensitivity index) per cell for the whole-brain analysis pipeline.

d′ measures how much the phasic fluorescence amplitude changes between
baseline and drug epochs, normalized by the pooled within-epoch variability:

    d′_i = (μ_drug - μ_baseline) / sqrt((σ²_baseline + σ²_drug) / 2)

Three amplitude modes are supported:
    "raw" — signed F_phasic (recommended; default)
    "abs" — |F_phasic| (unsigned amplitude)
    "rms" — F_phasic² (energy proxy)

Outputs saved per fish under dir_analysis / proj_ID / expt_ID/:
    phasic_dprime_cells_{mode}.npy      (n_cells,)
    phasic_deltaMean_cells_{mode}.npy   (n_cells,)

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/dprime.py
"""

import gc
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from utils.data_io import fish_dir


# ============================================================
# CORE: D′ PER CELL
# ============================================================

def dprime_per_cell(
    Fphasic,
    idx_base,
    idx_drug,
    amplitude_mode="raw",
    var_floor=1e-4,
    eps_var=1e-8,
    clip_abs=None,
):
    """
    Compute d′ and Δmean per cell between baseline and drug epochs.

    Parameters
    ----------
    Fphasic : np.ndarray or memmap, shape (n_cells, T)
        F_phasic traces.
    idx_base : slice or array-like
        Time indices for the baseline epoch.
    idx_drug : slice or array-like
        Time indices for the drug epoch.
    amplitude_mode : str
        "raw" — use F_phasic directly (signed; default).
        "abs" — use |F_phasic| (unsigned amplitude).
        "rms" — use F_phasic² (energy proxy; delta_mean in squared units).
    var_floor : float
        Minimum variance floor to stabilize the denominator. Prevents
        blowup in quiet windows where both epochs have near-zero variance.
    eps_var : float
        Additional epsilon added inside sqrt for numerical safety.
    clip_abs : float or None
        If set, clip d′ to [-clip_abs, +clip_abs] per cell.

    Returns
    -------
    dprime : np.ndarray, shape (n_cells,), float32
        Sensitivity index per cell. NaN for cells with degenerate windows.
    delta_mean : np.ndarray, shape (n_cells,), float32
        Raw mean difference (μ_drug - μ_baseline) in the amplitude domain.
    """
    # amplitude transform
    A = np.asarray(Fphasic, dtype=np.float32)
    if amplitude_mode == "raw":
        pass
    elif amplitude_mode == "abs":
        A = np.abs(A)
    elif amplitude_mode == "rms":
        A = A ** 2
    else:
        raise ValueError(f"amplitude_mode must be 'raw', 'abs', or 'rms', got {amplitude_mode!r}")

    base = A[:, idx_base]
    drug = A[:, idx_drug]

    if base.shape[1] < 2 or drug.shape[1] < 2:
        nan = np.full(A.shape[0], np.nan, dtype=np.float32)
        return nan, nan.copy()

    mu_b   = np.nanmean(base, axis=1)
    mu_d   = np.nanmean(drug, axis=1)
    var_b  = np.maximum(np.nanvar(base, axis=1, ddof=1), var_floor)
    var_d  = np.maximum(np.nanvar(drug, axis=1, ddof=1), var_floor)
    denom  = np.sqrt((var_b + var_d) / 2.0 + eps_var).astype(np.float32)

    dprime     = ((mu_d - mu_b) / denom).astype(np.float32)
    delta_mean = (mu_d - mu_b).astype(np.float32)

    dprime[~np.isfinite(dprime)] = np.nan
    dprime[denom <= 0]           = np.nan

    if clip_abs is not None:
        dprime = np.clip(dprime, -float(clip_abs), float(clip_abs))

    return dprime, delta_mean


def summarize_fish_dprime(dprime_cells):
    """
    Compute summary statistics for one fish's d′ distribution.

    Returns
    -------
    dict with keys: mean, median, p5, p25, p75, p95, p99, n_cells
    """
    x = dprime_cells[np.isfinite(dprime_cells)]
    if x.size == 0:
        return {k: np.nan for k in ("mean", "median", "p5", "p25", "p75", "p95", "p99")} | \
               {"n_cells": 0}
    p5, p25, p75, p95, p99 = np.nanpercentile(x, [5, 25, 75, 95, 99])
    return {
        "mean":    float(np.nanmean(x)),
        "median":  float(np.nanmedian(x)),
        "p5":      float(p5),
        "p25":     float(p25),
        "p75":     float(p75),
        "p95":     float(p95),
        "p99":     float(p99),
        "n_cells": int(x.size),
    }


# ============================================================
# PER-FISH RUNNER
# ============================================================

def _get_window(start_full, end_full, sr_hz, offset_sec):
    """Apply a frame offset to a window (e.g. skip early baseline frames)."""
    off = int(round(offset_sec * sr_hz))
    s   = int(start_full + off)
    e   = int(end_full)
    if e <= s:
        raise ValueError(f"Window is empty after offset: start={s}, end={e}")
    return s, e


def dprime_one_fish(
    fish,
    dir_analysis,
    baseline_start,
    baseline_end,
    drug_start,
    drug_end,
    sampling_rate_hz=1.0,
    offset_sec=15 * 60,
    amplitude_mode="raw",
    var_floor=1e-4,
    eps_var=1e-8,
    clip_abs=None,
    overwrite=False,
):
    """
    Compute and save d′ per cell for one fish.

    The offset_sec parameter skips the first `offset_sec` of each window
    (e.g. 15 min) to avoid edge artefacts from F_phasic decomposition.

    Reads  : dir_analysis / proj_ID / expt_ID / data_array_f_phasic.npy
    Writes : dir_analysis / proj_ID / expt_ID /
                phasic_dprime_cells_{amplitude_mode}.npy
                phasic_deltaMean_cells_{amplitude_mode}.npy

    Parameters
    ----------
    fish : tuple of (str, str)
    dir_analysis : str
    baseline_start, baseline_end : int
        Frame indices from config (before offset is applied).
    drug_start, drug_end : int
        Frame indices from config (before offset is applied).
    sampling_rate_hz : float
    offset_sec : float
        Seconds to skip at the start of each window. Default 15 min.
    amplitude_mode : str
        "raw", "abs", or "rms".
    overwrite : bool

    Returns
    -------
    dict with fish, status, and summary stats.
    """
    proj_ID, expt_ID = fish
    out_dir = fish_dir(dir_analysis, fish)

    dprime_path = out_dir / f"phasic_dprime_cells_{amplitude_mode}.npy"
    dmean_path  = out_dir / f"phasic_deltaMean_cells_{amplitude_mode}.npy"

    if not overwrite and dprime_path.exists() and dmean_path.exists():
        print(f"⏩ {expt_ID}: d′ exists, skipping.")
        return {"fish": fish, "status": "skipped"}

    Fp_path = out_dir / "f_phasic.npy"
    if not Fp_path.exists():
        raise FileNotFoundError(f"Missing F_phasic for {expt_ID}: {Fp_path}")

    Fp = np.load(str(Fp_path), mmap_mode="r")
    n_cells, Tfull = Fp.shape

    b_s, b_e = _get_window(baseline_start, baseline_end, sampling_rate_hz, offset_sec)
    d_s, d_e = _get_window(drug_start,     drug_end,     sampling_rate_hz, offset_sec)

    if b_e > Tfull or d_e > Tfull:
        raise ValueError(
            f"{expt_ID}: window exceeds Tfull={Tfull} "
            f"(baseline [{b_s},{b_e}), drug [{d_s},{d_e}))"
        )

    dprime_cells, delta_mean_cells = dprime_per_cell(
        Fp,
        idx_base=slice(b_s, b_e),
        idx_drug=slice(d_s, d_e),
        amplitude_mode=amplitude_mode,
        var_floor=var_floor,
        eps_var=eps_var,
        clip_abs=clip_abs,
    )

    np.save(str(dprime_path), dprime_cells)
    np.save(str(dmean_path),  delta_mean_cells)

    summ = summarize_fish_dprime(dprime_cells)

    print(
        f"✅ {expt_ID} | mean d′={summ['mean']:.3f} median={summ['median']:.3f} "
        f"p95={summ['p95']:.3f} n={summ['n_cells']}"
    )

    del Fp, dprime_cells, delta_mean_cells
    gc.collect()

    return {"fish": fish, "status": "ok", **summ}


# ============================================================
# IAAFT NULL FOR d′
# ============================================================
# Design: unlike the tonic GLM (which has a shared regressor to phase-
# randomize), d′ has no regressor — it's a direct baseline-vs-drug
# comparison on each cell's own trace. To keep the same "randomize one
# shared object, reuse across many cells" compute structure as the GLM
# null, we IAAFT-randomize the *window-label assignment* (which frames
# count as baseline vs drug) rather than each cell's signal. Every
# cell's real trace is left untouched; only which frames are called
# "baseline" and "drug" gets scrambled per surrogate.
#
# The label vector is binary (0=baseline, 1=drug) over the concatenated
# [baseline_frames, drug_frames] pool. IAAFT preserves the exact 0/1
# counts (rank-matching step) while approximately preserving the
# autocorrelation structure of the block assignment, then scrambles
# the timing — so the surrogate's baseline/drug windows are the same
# sizes as the real windows but disconnected from the true drug onset.

def _iaaft_1d(x, n_iter=50, rng=None):
    """
    Standard IAAFT surrogate of a 1D signal: preserves the amplitude
    (value) distribution exactly and approximately preserves the power
    spectrum, while randomizing phase/timing.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    x_sorted = np.sort(x)
    amp_spec = np.abs(np.fft.rfft(x))

    surrogate = rng.permutation(x)
    for _ in range(n_iter):
        f = np.fft.rfft(surrogate)
        phases = np.angle(f)
        surrogate = np.fft.irfft(amp_spec * np.exp(1j * phases), n=n)
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = x_sorted[ranks]
    return surrogate


def iaaft_null_dprime_one_fish(
    fish,
    dir_analysis,
    baseline_start,
    baseline_end,
    drug_start,
    drug_end,
    sampling_rate_hz=1.0,
    offset_sec=15 * 60,
    amplitude_mode="raw",
    var_floor=1e-4,
    eps_var=1e-8,
    n_surrogates=200,
    n_iter_iaaft=50,
    n_cells_null=20000,
    null_percentile=95,
    seed=0,
    overwrite=False,
):
    """
    Build an IAAFT-based null distribution for phasic d′ and save a
    single symmetric magnitude threshold (matches notebook convention
    of a symmetric ±thresh, e.g. ±0.5).

    Writes to dir_analysis / proj_ID / expt_ID /:
        phasic_dprime_iaaft_nullp{P}_thresh.npy   (scalar, float32)
        phasic_dprime_null__iaaft.npy             (n_surrogates, n_sample)

    Returns
    -------
    dict with fish, status, thresh.
    """
    proj_ID, expt_ID = fish
    out_dir = fish_dir(dir_analysis, fish)

    ptag = int(null_percentile)
    thresh_path = out_dir / f"phasic_dprime_iaaft_nullp{ptag}_thresh.npy"
    null_path   = out_dir / "phasic_dprime_null__iaaft.npy"

    if not overwrite and thresh_path.exists():
        thr = float(np.load(str(thresh_path)))
        print(f"⏩ {expt_ID}: d′ IAAFT null exists, thr=±{thr:.4f}")
        return {"fish": fish, "status": "skipped", "thresh": thr}

    Fp_path = out_dir / "f_phasic.npy"
    if not Fp_path.exists():
        raise FileNotFoundError(f"Missing F_phasic for {expt_ID}: {Fp_path}")

    Fp = np.load(str(Fp_path), mmap_mode="r")
    n_cells, Tfull = Fp.shape

    b_s, b_e = _get_window(baseline_start, baseline_end, sampling_rate_hz, offset_sec)
    d_s, d_e = _get_window(drug_start,     drug_end,     sampling_rate_hz, offset_sec)

    if d_s < b_e:
        raise ValueError(
            f"{expt_ID}: drug window starts before baseline window ends "
            f"(unexpected overlap) — b_e={b_e}, d_s={d_s}"
        )

    n_base = b_e - b_s
    n_drug = d_e - d_s
    label  = np.concatenate([np.zeros(n_base), np.ones(n_drug)])

    # ── sample cells for the null ──────────────────────────────────────
    rng = np.random.default_rng(seed)
    n_sample = min(n_cells_null, n_cells)
    sample_idx = np.sort(rng.choice(n_cells, size=n_sample, replace=False))

    # load only the contiguous [b_s, d_e) block for sampled cells
    Fp_block = np.asarray(Fp[sample_idx, b_s:d_e], dtype=np.float32)

    base_rel   = np.arange(0, n_base)
    drug_rel   = np.arange(d_s - b_s, d_s - b_s + n_drug)
    global_rel = np.concatenate([base_rel, drug_rel])   # order matches `label`

    null_vals = np.empty((n_surrogates, n_sample), dtype=np.float32)

    iterator = tqdm(range(n_surrogates), desc=f"{expt_ID} d′ IAAFT null", unit="surr")
    for s in iterator:
        surr_label = _iaaft_1d(label, n_iter=n_iter_iaaft, rng=rng)
        order = np.argsort(surr_label)                  # low → high
        surr_base_rel = global_rel[order[:n_base]]
        surr_drug_rel = global_rel[order[n_base:]]

        dprime_s, _ = dprime_per_cell(
            Fp_block,
            idx_base=surr_base_rel,
            idx_drug=surr_drug_rel,
            amplitude_mode=amplitude_mode,
            var_floor=var_floor,
            eps_var=eps_var,
            clip_abs=None,
        )
        null_vals[s] = dprime_s

    null_flat = null_vals.ravel()
    null_flat = null_flat[np.isfinite(null_flat)]
    thr = float(np.percentile(np.abs(null_flat), null_percentile)) if null_flat.size else np.nan

    np.save(str(thresh_path), np.array(thr, dtype=np.float32))
    np.save(str(null_path),   null_vals.astype(np.float32))

    print(f"✅ {expt_ID}: d′ IAAFT null done | n_surr={n_surrogates} "
          f"n_cells={n_sample} | thr(p{ptag})=±{thr:.4f}")

    del Fp, Fp_block, null_vals
    gc.collect()

    return {"fish": fish, "status": "ok", "thresh": thr}


# ============================================================
# PHASIC RESPONDER INDICES (threshold real d′ against IAAFT null)
# ============================================================

def save_dprime_responder_idx(
    fish,
    dir_analysis,
    amplitude_mode="raw",
    null_percentile=95,
    external_thr=None,
    overwrite=True,
):
    """
    Classify cells as positive/negative phasic responders by comparing
    each cell's real d′ against a symmetric threshold.

    If external_thr is provided (e.g. a ctrl-pooled IAAFT threshold),
    that value is used directly and the per-fish threshold file is ignored.
    Otherwise the per-fish IAAFT null threshold file is loaded from disk.

    Writes to dir_analysis / proj_ID / expt_ID /:
        phasic_pos_dprime_iaaft_nullp{P}_idxs.npy
        phasic_neg_dprime_iaaft_nullp{P}_idxs.npy

    Returns
    -------
    dict with fish, status, n_pos, n_neg, thresh.
    """
    proj_ID, expt_ID = fish
    out_dir = fish_dir(dir_analysis, fish)
    ptag = int(null_percentile)

    dprime_path = out_dir / f"phasic_dprime_cells_{amplitude_mode}.npy"
    pos_path    = out_dir / f"phasic_pos_dprime_iaaft_nullp{ptag}_idxs.npy"
    neg_path    = out_dir / f"phasic_neg_dprime_iaaft_nullp{ptag}_idxs.npy"

    if not overwrite and pos_path.exists() and neg_path.exists():
        print(f"⏩ {expt_ID}: d′ responder idx exist, skipping.")
        return {"fish": fish, "status": "skipped"}

    if not dprime_path.exists():
        raise FileNotFoundError(f"Missing d′ cells for {expt_ID}: {dprime_path}")

    if external_thr is not None:
        thr = float(external_thr)
    else:
        thresh_path = out_dir / f"phasic_dprime_iaaft_nullp{ptag}_thresh.npy"
        if not thresh_path.exists():
            raise FileNotFoundError(
                f"Missing IAAFT threshold for {expt_ID}: {thresh_path}\n"
                "Run iaaft_null_dprime_one_fish() first, or pass external_thr."
            )
        thr = float(np.load(str(thresh_path)))

    dprime_cells = np.load(str(dprime_path))
    pos_idx = np.where(dprime_cells > thr)[0].astype(np.int64)
    neg_idx = np.where(dprime_cells < -thr)[0].astype(np.int64)

    np.save(str(pos_path), pos_idx)
    np.save(str(neg_path), neg_idx)

    n_cells = dprime_cells.size
    print(
        f"✅ {expt_ID}: d′ responders | thr=±{thr:.4f} | "
        f"POS {pos_idx.size} ({pos_idx.size/n_cells*100:.2f}%) "
        f"NEG {neg_idx.size} ({neg_idx.size/n_cells*100:.2f}%)"
    )

    return {
        "fish": fish, "status": "ok",
        "n_pos": int(pos_idx.size), "n_neg": int(neg_idx.size),
        "thresh": thr,
    }


# ============================================================
# CTRL-POOLED IAAFT THRESHOLD
# ============================================================

def compute_ctrl_pooled_threshold(ctrl_fish_list, dir_analysis, null_percentile=95):
    """
    Pool surrogate d′ values across all ctrl fish null distributions and
    return the p{null_percentile} of |d′_null| as a single scalar threshold.

    This threshold is used for group-comparison figures so that both ctrl
    and expt fish are classified against a common ctrl-level noise bar,
    rather than each fish's own per-fish noise floor.

    Reads: fish_dir(dir_analysis, fish) / phasic_dprime_null__iaaft.npy
           shape: (n_surrogates, n_cells_sampled)

    Parameters
    ----------
    ctrl_fish_list  : list of (proj_ID, expt_ID) tuples
    dir_analysis    : str or Path
    null_percentile : int, default 95

    Returns
    -------
    thr : float
        Symmetric threshold; classify as responder if |d′| > thr.
    """
    all_null_vals = []
    for fish in ctrl_fish_list:
        null_path = fish_dir(dir_analysis, fish) / "phasic_dprime_null__iaaft.npy"
        if not null_path.exists():
            print(f"  ⚠️  {fish[1]}: null distribution missing — skipping from pool")
            continue
        null_arr = np.load(str(null_path))      # (n_surr, n_cells_sampled)
        flat = null_arr.ravel()
        flat = flat[np.isfinite(flat)]
        all_null_vals.append(flat)
        print(f"  📦 {fish[1]}: {flat.size:,} surrogate d′ values loaded")

    if not all_null_vals:
        raise RuntimeError(
            "No ctrl fish null distributions found — "
            "run iaaft_null_dprime_one_fish() for ctrl fish first."
        )

    pooled = np.concatenate(all_null_vals)
    thr = float(np.percentile(np.abs(pooled), null_percentile))
    print(
        f"\n  ✅ Ctrl-pooled threshold (p{null_percentile}): ±{thr:.4f}  "
        f"({len(all_null_vals)}/{len(ctrl_fish_list)} ctrl fish, "
        f"{pooled.size:,} surrogate values)"
    )
    return thr
