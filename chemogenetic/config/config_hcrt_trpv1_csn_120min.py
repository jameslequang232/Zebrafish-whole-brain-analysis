"""
hcrt_trpv1_csn_120min.py
=========================
Config for the HCRT-TRPV1 capsaicin 120-minute chemogenetic experiment.

Single source of truth for paths, parameters, and fish lists.
Everything downstream (modules + run codes) imports from here.

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/config/hcrt_trpv1_csn_120min.py
"""

# ============================================================
# BASE PATHS  (three distinct trees)
# ============================================================
# Tree 1: voluseg raw input/output (read-only).
#         read_data() builds: dir_voluseg / proj_ID / expt_ID / output/
dir_voluseg = "/resnick/groups/Proberlab/yun/lightsheet/"

# Tree 2: ANTs registration products + extracted cell coords.
dir_registration = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/registration/"

# Tree 3: analysis products (F_tonic, GLM, d', figures). Pipeline WRITES here.
dir_analysis = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/chemogenetic/"

PYTHON_BIN = "/resnick/home/ychiu/miniconda3/envs/voluseg/bin/python"

# ============================================================
# PROJECT FOLDERS
# ============================================================
CTRL_PROJ = "huc-h2b-g8m_csn_120min"
EXPT_PROJ = "hcrt-trpv1_huc-h2b-g8m_csn_120min"


# ============================================================
# FISH LISTS  — each fish is a (proj_ID, expt_ID) tuple
# ============================================================
ctrl_fish = [
    (CTRL_PROJ, "251021_huc-h2b-g8m_csn_10uM_fish1"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish1"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish2"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish3"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish4"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish5"),
    (CTRL_PROJ, "251126_huc-h2b-g8m_csn_10uM_fish1"),
]   # N = 7 

expt_fish = [
    (EXPT_PROJ, "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4"),
    (EXPT_PROJ, "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (EXPT_PROJ, "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (EXPT_PROJ, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (EXPT_PROJ, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (EXPT_PROJ, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3"),
    (EXPT_PROJ, "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (EXPT_PROJ, "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (EXPT_PROJ, "260515_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
]   # N = 9

all_fish = ctrl_fish + expt_fish


# ============================================================
# IMAGING SPECS
# ============================================================
sec_per_volume = 1
volume_per_sec = 1
sampling_rate_hz = volume_per_sec

n_slices = 40
depth = 250                            # microns
binning = 1
res_x = 1.52 * binning
res_y = 1.52 * binning
res_z = depth / n_slices
rotation_k = 2                         # 180° rotation for mapzebrain alignment


# ============================================================
# DRUG PERFUSION  (CSTR model: dC/dt = (Q/V)(Cin - C))
# ============================================================
drug_uM = 10.0
V_ml = 15.0
Q_ml_min = 4

# All fish use 10 µM capsaicin — single global drug_uM applies to all.

# Epoch windows in FRAMES (minutes * 60 * volume_per_sec)
baseline_start = 0  * 60 * volume_per_sec
baseline_end   = 45 * 60 * volume_per_sec

drug_start = 45 * 60 * volume_per_sec
drug_end   = 90 * 60 * volume_per_sec

wash_start = 90  * 60 * volume_per_sec
wash_end   = 120 * 60 * volume_per_sec


# ============================================================
# DECOMPOSITION  (F → F_tonic + F_phasic)
# ============================================================
df_f_percentile = 20
f_tonic_window_size = 600              # seconds
f_tonic_percentile  = 20


# ============================================================
# PERMUTATION TEST + BH-FDR
# ============================================================
p_thresh_permutation  = 0.005
n_resample_permutation = 500
BH_Q = 0.05


# ============================================================
# TONIC GLM
# ============================================================
input_tag    = "C"                     # "C" or "dC"
K_global     = 600                     # kernel length (frames)
drift_global = 1                       # polynomial drift order
lam_global   = 0.5                     # ridge penalty
lag_global   = 0                       # causal lag (frames)

INCLUDED_BASELINE = 45.0               # minutes of baseline in GLM window
CLIP_ABS_DZ       = 50.0               # |ΔZ| clipping threshold
NULL_TAG          = "iaaft"            # "iaaft" or "shift"
RESPONDER_NULL_THRESH = 99             # null percentile for responder cutoff
L_MIN = 10.0                           # plateau duration (minutes) for plateau-ΔZ

# Derived: subfolder name for all GLM outputs (defined once, used everywhere)
param_folder_name = (
    f"in{input_tag}_K{K_global}_drift{drift_global}_lam{lam_global}_lag{lag_global}"
)


# ============================================================
# GROUP TAGS + PLOT META
# ============================================================
EXPT_TAG = "HCRT-TRPV1"
CTRL_TAG = "CTRL"

PLOT_META = {
    EXPT_TAG: {"label": EXPT_TAG, "color": "indianred", "alpha": 0.7},
    CTRL_TAG: {"label": CTRL_TAG, "color": "grey",      "alpha": 0.7},
}

GROUP_OF_FISH = {fish: CTRL_TAG for fish in ctrl_fish}
GROUP_OF_FISH.update({fish: EXPT_TAG for fish in expt_fish})


def get_group_meta(fish):
    """Return (group_tag, plot_meta_dict) for a fish tuple."""
    tag = GROUP_OF_FISH[fish]
    return tag, PLOT_META[tag]


# ============================================================
# COMPARISON TAG + FIGURE PATH
# ============================================================
COMPARISON_TAG = f"{EXPT_TAG}_vs_{CTRL_TAG}"   # e.g. "HCRT-TRPV1_vs_CTRL"


def comparison_fig_dir(dir_analysis, comparison_tag=None):
    """
    Return the figure directory for a group-level comparison.

    Parameters
    ----------
    dir_analysis : str
    comparison_tag : str or None
        Defaults to COMPARISON_TAG from this config.

    Returns
    -------
    Path
        dir_analysis / comparisons / comparison_tag / figures
    """
    from pathlib import Path
    tag = comparison_tag if comparison_tag is not None else COMPARISON_TAG
    p = Path(dir_analysis) / "comparisons" / tag / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
