"""
data_io.py
==========
Path resolution, data loading, and folder management for the analysis pipeline.

Every module resolves fish paths through fish_dir() — this is the single place
where (proj_ID, expt_ID) tuples get turned into real filesystem paths.

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/io/data_io.py
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile as tiff


# ============================================================
# PATH RESOLUTION
# ============================================================

def fish_dir(base_dir, fish):
    """
    Resolve a (proj_ID, expt_ID) tuple to an absolute path under base_dir.

    Parameters
    ----------
    base_dir : str or Path
        One of dir_voluseg, dir_registration, or dir_analysis from config.
    fish : tuple of (str, str)
        (proj_ID, expt_ID).

    Returns
    -------
    Path
        base_dir / proj_ID / expt_ID
    """
    proj_ID, expt_ID = fish
    return Path(base_dir) / proj_ID / expt_ID


# ============================================================
# DATA LOADING
# ============================================================

def read_data(fish, dir_voluseg):
    """
    Load voluseg-processed timeseries, cell coordinates, and volume mean.

    Reads from: dir_voluseg / proj_ID / expt_ID / output /
        - cells0_clean.hdf5  (cell_timeseries, cell_x, cell_y, cell_z)
        - volume0.hdf5       (volume_mean)

    Parameters
    ----------
    fish : tuple of (str, str)
        (proj_ID, expt_ID).
    dir_voluseg : str
        Base path to voluseg data tree (config.dir_voluseg).

    Returns
    -------
    data_np : np.ndarray
        Cell timeseries, shape (n_cells, n_timepoints).
    volume_mean : np.ndarray
        Mean volume, transposed to (X, Y, Z).
    cell_x, cell_y, cell_z : np.ndarray
        Cell coordinates. NOTE: voluseg's cell_y maps to conceptual X,
        and voluseg's cell_x maps to conceptual Y (axis swap applied here).
    """
    output_dir = fish_dir(dir_voluseg, fish) / "output"

    # --- cell timeseries and coordinates ---
    cells_path = output_dir / "cells0_clean.hdf5"
    with h5py.File(str(cells_path), "r") as cells:
        data_np = pd.DataFrame(cells["cell_timeseries"]).to_numpy()
        # voluseg axis convention swap:
        cell_x = np.array(cells["cell_y"])   # voluseg cell_y → conceptual X
        cell_y = np.array(cells["cell_x"])   # voluseg cell_x → conceptual Y
        cell_z = np.array(cells["cell_z"])

    print(
        f"cell_x range: {np.min(cell_x), np.max(cell_x)}, "
        f"cell_y range: {np.min(cell_y), np.max(cell_y)}, "
        f"cell_z range: {np.min(cell_z), np.max(cell_z)}"
    )

    # --- volume mean ---
    volume_path = output_dir / "volume0.hdf5"
    with h5py.File(str(volume_path), "r") as volume:
        volume_mean = volume["volume_mean"][()]

    print(f"Loaded raw hdf5 data as numpy array, shape: {volume_mean.shape}")
    volume_mean = np.transpose(volume_mean, axes=(1, 2, 0))
    print(f"Transposed data numpy array shape into (X, Y, Z): {volume_mean.shape}")

    return data_np, volume_mean, cell_x, cell_y, cell_z


def get_regions(output_dir):
    """
    Load all .tif region masks from a directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing .tif region mask files.

    Returns
    -------
    arrays : list of np.ndarray
        Each transposed to (X, Y, Z).
    file_names : list of str
        Corresponding filenames.
    """
    output_dir = Path(output_dir)
    arrays = []
    file_names = []

    for fname in sorted(output_dir.iterdir()):
        if fname.suffix == ".tif":
            image_data = tiff.imread(str(fname))
            image_array = np.transpose(image_data, axes=(1, 2, 0))
            arrays.append(image_array)
            file_names.append(fname.name)

    return arrays, file_names


# ============================================================
# FOLDER MANAGEMENT
# ============================================================

def ensure_fish_dir(base_dir, fish):
    """
    Create proj_ID/expt_ID directories under base_dir if they don't exist.

    Parameters
    ----------
    base_dir : str or Path
        One of the three config base paths.
    fish : tuple of (str, str)
        (proj_ID, expt_ID).

    Returns
    -------
    Path
        The created (or existing) directory path.
    """
    path = fish_dir(base_dir, fish)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_subfolder(base_dir, fish, subfolder):
    """
    Create a named subfolder (e.g. 'glm', 'figures') under a fish's directory.

    Parameters
    ----------
    base_dir : str or Path
        Base path (typically dir_analysis).
    fish : tuple of (str, str)
        (proj_ID, expt_ID).
    subfolder : str
        Name of the subfolder to create (e.g. "glm", "figures").

    Returns
    -------
    Path
        The created (or existing) subfolder path.
    """
    path = fish_dir(base_dir, fish) / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# NPY CONVENIENCE
# ============================================================

def save_npy(base_dir, fish, filename, array):
    """Save an array to dir_analysis/proj_ID/expt_ID/filename."""
    path = fish_dir(base_dir, fish) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
    print(f"Saved {path}  shape={array.shape}")


def load_npy(base_dir, fish, filename):
    """Load an array from base_dir/proj_ID/expt_ID/filename."""
    path = fish_dir(base_dir, fish) / filename
    arr = np.load(str(path))
    print(f"Loaded {path}  shape={arr.shape}")
    return arr


# ============================================================
# FIGURE PATH HELPERS
# ============================================================

def fish_fig_dir(dir_analysis, fish):
    """
    Per-fish figure directory.

    Returns
    -------
    Path
        dir_analysis / proj_ID / expt_ID / figures
    """
    p = fish_dir(dir_analysis, fish) / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def comparison_fig_dir(dir_analysis, comparison_tag):
    """
    Group-level comparison figure directory.

    Parameters
    ----------
    dir_analysis : str
    comparison_tag : str
        e.g. "HCRT-TRPV1_vs_CTRL" — from config.COMPARISON_TAG

    Returns
    -------
    Path
        dir_analysis / comparisons / comparison_tag / figures
    """
    p = Path(dir_analysis) / "comparisons" / comparison_tag / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ============================================================
# TIMESTAMP LOGGING
# ============================================================

def log_step(msg):
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)