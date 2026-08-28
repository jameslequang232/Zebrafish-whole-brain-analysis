"""
config_hcrt_trpv1_pooled_csn_120min.py
=======================================
Config for pooled HCRT-TRPV1 cohort: main (N=9) + inj (N=3) = N=12.

Purpose:
    Exploratory brain map only — not for publication without caveat.
    Inj fish have double GCaMP in HCRT cells (huc-h2b-g8m + hcrt-h2b-g8m),
    which inflates HCRT-voxel ΔZ relative to the main cohort. Pooling
    maximises statistical power for identifying downstream HCRT targets.

Location:
    ~/zwba/chemogenetic/config/config_hcrt_trpv1_pooled_csn_120min.py
"""

# ============================================================
# BASE PATHS
# ============================================================
dir_voluseg      = "/resnick/groups/Proberlab/yun/lightsheet/"
dir_registration = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/registration/"
dir_analysis     = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/chemogenetic/"

PYTHON_BIN = "/resnick/home/ychiu/miniconda3/envs/voluseg/bin/python"

# ============================================================
# PROJECT FOLDERS
# ============================================================
EXPT_PROJ = "hcrt-trpv1_huc-h2b-g8m_csn_120min"
INJ_PROJ  = "hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_120min"
CTRL_PROJ = "huc-h2b-g8m_csn_120min"

# ============================================================
# FISH LISTS
# ============================================================

# main cohort (N=9)
expt_fish_csn = [
    (EXPT_PROJ, "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4"),
    (EXPT_PROJ, "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (EXPT_PROJ, "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (EXPT_PROJ, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (EXPT_PROJ, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (EXPT_PROJ, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3"),
    (EXPT_PROJ, "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (EXPT_PROJ, "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (EXPT_PROJ, "260515_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
]  # N = 9

# inj cohort (N=3)
expt_fish_csn_inj = [
    (INJ_PROJ, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish1"),
    (INJ_PROJ, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish2"),
    (INJ_PROJ, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish3"),
]  # N = 3

# pooled expt (N=12) — alias expected by run_temporal_intensity_map.py
expt_fish = expt_fish_csn + expt_fish_csn_inj   # N = 12

# ctrl group — same huc-h2b-g8m fish used across all CSN comparisons
ctrl_fish = [
    (CTRL_PROJ, "251021_huc-h2b-g8m_csn_10uM_fish1"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish1"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish2"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish3"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish4"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish5"),
    (CTRL_PROJ, "251126_huc-h2b-g8m_csn_10uM_fish1"),
]  # N = 7

all_fish = expt_fish   # used by run_decompose / run_medoids (already done)


# ============================================================
# IMAGING SPECS
# ============================================================
sec_per_volume   = 1
volume_per_sec   = 1
sampling_rate_hz = volume_per_sec

n_slices   = 40
depth      = 250
binning    = 1
res_x      = 1.52 * binning
res_y      = 1.52 * binning
res_z      = depth / n_slices
rotation_k = 2


# ============================================================
# DRUG PERFUSION
# ============================================================
drug_uM  = 10.0
V_ml     = 15.0
Q_ml_min = 4

baseline_start = 0  * 60 * volume_per_sec
baseline_end   = 45 * 60 * volume_per_sec
drug_start     = 45 * 60 * volume_per_sec
drug_end       = 90 * 60 * volume_per_sec
wash_start     = 90  * 60 * volume_per_sec
wash_end       = 120 * 60 * volume_per_sec


# ============================================================
# DECOMPOSITION
# ============================================================
df_f_percentile     = 20
f_tonic_window_size = 600
f_tonic_percentile  = 20


# ============================================================
# PERMUTATION TEST + BH-FDR
# ============================================================
p_thresh_permutation   = 0.005
n_resample_permutation = 500
BH_Q = 0.05


# ============================================================
# TONIC GLM
# ============================================================
input_tag    = "C"
K_global     = 600
drift_global = 1
lam_global   = 0.5
lag_global   = 0

INCLUDED_BASELINE     = 45.0
CLIP_ABS_DZ           = 50.0
NULL_TAG              = "iaaft"
RESPONDER_NULL_THRESH = 99
L_MIN                 =10.0

param_folder_name = (
    f"in{input_tag}_K{K_global}_drift{drift_global}_lam{lam_global}_lag{lag_global}"
)


# ============================================================
# GROUP TAGS + PLOT META
# ============================================================
EXPT_TAG = "HCRT-TRPV1 (pooled)"
CTRL_TAG = "CTRL"

PLOT_META = {
    EXPT_TAG: {"label": EXPT_TAG, "color": "indianred", "alpha": 0.7},
    CTRL_TAG: {"label": CTRL_TAG, "color": "grey",      "alpha": 0.7},
}

GROUP_OF_FISH = {fish: EXPT_TAG for fish in expt_fish}
GROUP_OF_FISH.update({fish: CTRL_TAG for fish in ctrl_fish})


def get_group_meta(fish):
    tag = GROUP_OF_FISH[fish]
    return tag, PLOT_META[tag]


# ============================================================
# COMPARISON TAG + FIGURE PATH
# ============================================================
COMPARISON_TAG = "HCRT-TRPV1-pooled_vs_CTRL"   # N=12 vs N=7


def comparison_fig_dir(dir_analysis, comparison_tag=None):
    from pathlib import Path
    tag = comparison_tag if comparison_tag is not None else COMPARISON_TAG
    p = Path(dir_analysis) / "comparisons" / tag / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
