"""
run_glm.py
==========
SLURM entry point: run the full GLM pipeline for all fish.

Stages:
    A. CV            — per-fish hyperparameter selection
    B. Choose global — aggregate CV results
    C. Refit         — refit all cells with global params
    D. Ablation      — drift-only ΔR²
    E. IAAFT null    — surrogate null distribution (now parallel across fish;
                        BLAS threads capped per worker to avoid oversubscription)
    F. Responders    — threshold + save pos/neg idx

Regressor: empirical HCRT population trace from hcrt_all.csv
           (loaded once, passed as u_ext to all stages).
           Falls back to CSTR capsaicin model if CSV not found.

Usage:
    sbatch submit_glm.sh --config config_hcrt_trpv1_csn_120min

Location:
    ~/zwba/chemogenetic/run/run_glm.py
"""

import argparse
import os
import sys
from pathlib import Path

# ── cap BLAS threads per worker BEFORE numpy import ─────────────────────────
# With N_JOBS_STAGE_E parallel fish workers each doing ridge regression,
# uncapped BLAS threading causes massive core oversubscription and can be
# SLOWER than serial. Set so N_JOBS_STAGE_E * BLAS_THREADS_PER_WORKER ≈
# total allocated CPUs (see submit_glm.sh --cpus-per-task).
BLAS_THREADS_PER_WORKER = "4"
os.environ["OMP_NUM_THREADS"]        = BLAS_THREADS_PER_WORKER
os.environ["MKL_NUM_THREADS"]        = BLAS_THREADS_PER_WORKER
os.environ["OPENBLAS_NUM_THREADS"]   = BLAS_THREADS_PER_WORKER
os.environ["NUMEXPR_NUM_THREADS"]    = BLAS_THREADS_PER_WORKER
os.environ["VECLIB_MAXIMUM_THREADS"] = BLAS_THREADS_PER_WORKER

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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

all_fish         = cfg.all_fish
dir_analysis     = cfg.dir_analysis
drug_end         = cfg.drug_end
drug_start       = cfg.drug_start
drug_uM          = cfg.drug_uM
INCLUDED_BASELINE = cfg.INCLUDED_BASELINE
input_tag        = cfg.input_tag
K_global         = cfg.K_global
drift_global     = cfg.drift_global
lam_global       = cfg.lam_global
lag_global       = cfg.lag_global
NULL_TAG         = cfg.NULL_TAG
param_folder_name = cfg.param_folder_name
Q_ml_min         = cfg.Q_ml_min
RESPONDER_NULL_THRESH = 95 #cfg.RESPONDER_NULL_THRESH
sampling_rate_hz = cfg.sampling_rate_hz
V_ml             = cfg.V_ml

from chemogenetic.glm import (
    choose_global_params,
    cv_one_fish,
    refit_one_fish,
    ablation_one_fish,
    iaaft_null_one_fish,
    save_responder_idx,
)

# ============================================================
# STAGE TOGGLES
# ============================================================
RUN_CV         = False   # set True for new experiments needing CV
RUN_REFIT      = False
RUN_ABLATION   = False
RUN_NULL       = False
RUN_RESPONDERS = True

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE_CV         = False
OVERWRITE_REFIT      = False
OVERWRITE_ABLATION   = False
OVERWRITE_NULL       = False
OVERWRITE_RESPONDERS = True

K_LIST        = (60, 120, 300, 600, 900, 1200)
DRIFT_ORDERS  = (1, 2, 3)
LAM_LIST      = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)
BLOCK_LEN_SEC = 10 * 60
N_CELLS_CV    = 2000
CHUNK_CELLS   = 2000

N_SURROGATES  = 200
N_ITER_IAAFT  = 50
N_CELLS_NULL  = 20000

# control the sign classification in Stage F (save_responder_idx) — they're separate from the fit window
# define which minutes, relative to the start of the fit window, get averaged for this comparison.
BASELINE_WIN_MIN = (20.0, 45.0)
DRUG_WIN_MIN     = (60.0, 75.0)

N_JOBS_FISH = 16

# Stage E (IAAFT null) is now parallel across fish. Keep this modest since
# each worker also spawns BLAS_THREADS_PER_WORKER threads internally —
# N_JOBS_STAGE_E * int(BLAS_THREADS_PER_WORKER) should stay close to your
# allocated --cpus-per-task in submit_glm.sh (e.g. 8 * 4 = 32).
N_JOBS_STAGE_E = 8

FIT_BASELINE_SEC = INCLUDED_BASELINE * 60

# ============================================================
# HCRT EMPIRICAL REGRESSOR
# ============================================================
# Load empirical HCRT population trace from hcrt_all.csv files.
# Same pipeline as hcrt_regressor.ipynb: rolling 20th-pct tonic → z-score.
# Falls back to CSTR capsaicin model (u_ext=None) if files not found.

HCRT_BASE = "/resnick/home/ychiu/yun/lightsheet/hcrt-trpv1_hcrt-h2b-g8m_120min"
HCRT_FISH_PATHS = [
    f"{HCRT_BASE}/260426_hcrt-trpv1_hcrt-h2b-g8m_csn_10uM_fish1/hcrt_all.csv",
    f"{HCRT_BASE}/260426_hcrt-trpv1_hcrt-h2b-g8m_csn_10uM_fish2/hcrt_all.csv",
    f"{HCRT_BASE}/260426_hcrt-trpv1_hcrt-h2b-g8m_csn_10uM_fish3/hcrt_all.csv",
]
HCRT_TONIC_WINDOW = 600
HCRT_TONIC_PCTILE = 0.20
CSN_ONSET_VOL     = int(drug_start)
TOTAL_VOLS        = 7200

u_ext = None   # will be set below if CSV files exist

if all(Path(p).exists() for p in HCRT_FISH_PATHS):
    fish_tonic = []
    for path in HCRT_FISH_PATHS:
        raw = pd.read_csv(path)["Mean"].values
        ft  = (pd.Series(raw)
                 .rolling(HCRT_TONIC_WINDOW, center=True, min_periods=1)
                 .quantile(HCRT_TONIC_PCTILE)
                 .values)
        mu    = ft[:CSN_ONSET_VOL].mean()
        sigma = ft[:CSN_ONSET_VOL].std(ddof=1)
        fish_tonic.append((ft - mu) / max(sigma, 1e-6))

    u_ext = np.mean(fish_tonic, axis=0).astype(np.float32)
    print(f"✅ HCRT regressor loaded: mean across {len(fish_tonic)} fish, "
          f"peak={u_ext.max():.2f} at vol {u_ext.argmax()}")
    input_tag = "HCRT"   # update tag so output folder name reflects regressor
else:
    print("⚠️  HCRT csv files not found — falling back to CSTR capsaicin regressor")


# ============================================================
# HELPERS
# ============================================================
def _common_kwargs(fish):
    return dict(
        fish=fish,
        dir_analysis=dir_analysis,
        sampling_rate_hz=sampling_rate_hz,
        drug_start_frame_full=drug_start,
        drug_end_frame_full=drug_end,
        drug_uM=drug_uM,
        V_ml=V_ml,
        Q_ml_min=Q_ml_min,
        input_tag=input_tag,
        lag_global=lag_global,
        fit_baseline_sec=FIT_BASELINE_SEC,
        chunk_cells_fit=CHUNK_CELLS,
        u_ext=u_ext,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"GLM pipeline | {len(all_fish)} fish")
    print(f"  config={args.config}  input_tag={input_tag}")
    print(f"  K={K_global}  drift={drift_global}  lam={lam_global}  "
          f"lag={lag_global}  null={NULL_TAG}\n")

    # ── A: CV ──────────────────────────────────────────────────
    if RUN_CV:
        print("── Stage A: CV ─────────────────────────────────────")

        def _cv(fish):
            try:
                return cv_one_fish(
                    **_common_kwargs(fish),
                    K_list=K_LIST, drift_orders=DRIFT_ORDERS, lam_list=LAM_LIST,
                    block_len_sec=BLOCK_LEN_SEC, n_cells_cv=N_CELLS_CV,
                    overwrite=OVERWRITE_CV, show_progress=False, verbose_cv=False,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
            delayed(_cv)(fish) for fish in tqdm(all_fish, desc="CV")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    # ── B: Choose globals ──────────────────────────────────────
    print("\n── Stage B: Using config globals (CV disabled) ──────────")
    print(f"  K={K_global}  drift={drift_global}  lam={lam_global}")

    # ── C: Refit ───────────────────────────────────────────────
    if RUN_REFIT:
        print("\n── Stage C: Global refit ───────────────────────────────")

        def _refit(fish):
            try:
                return refit_one_fish(
                    **_common_kwargs(fish),
                    K_global=K_global, drift_global=drift_global, lam_global=lam_global,
                    overwrite=OVERWRITE_REFIT, show_progress=False,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
            delayed(_refit)(fish) for fish in tqdm(all_fish, desc="Refit")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    # ── D: Ablation ────────────────────────────────────────────
    if RUN_ABLATION:
        print("\n── Stage D: Drift-only ablation ────────────────────────")

        def _ablation(fish):
            try:
                return ablation_one_fish(
                    **_common_kwargs(fish),
                    K_global=K_global, drift_global=drift_global, lam_global=lam_global,
                    overwrite=OVERWRITE_ABLATION, show_progress=False,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
            delayed(_ablation)(fish) for fish in tqdm(all_fish, desc="Ablation")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    # ── E: IAAFT null ──────────────────────────────────────────
    if RUN_NULL:
        print(f"\n── Stage E: IAAFT null (parallel, n_jobs={N_JOBS_STAGE_E}, "
              f"{BLAS_THREADS_PER_WORKER} BLAS threads/worker) ──────────")

        def _null(fish):
            try:
                result = iaaft_null_one_fish(
                    **_common_kwargs(fish),
                    K_global=K_global, drift_global=drift_global, lam_global=lam_global,
                    n_surrogates=N_SURROGATES, n_iter_iaaft=N_ITER_IAAFT,
                    n_cells_null=N_CELLS_NULL,
                    null_percentile=RESPONDER_NULL_THRESH,
                    overwrite=OVERWRITE_NULL, show_progress=False,
                )
                return {"fish": fish, "status": result["status"]}
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_STAGE_E, backend="loky")(
            delayed(_null)(fish) for fish in tqdm(all_fish, desc="IAAFT null")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    # ── F: Responder indices ───────────────────────────────────
    if RUN_RESPONDERS:
        print("\n── Stage F: Save responder indices ─────────────────────")

        def _responders(fish):
            try:
                return save_responder_idx(
                    fish=fish,
                    dir_analysis=dir_analysis,
                    sampling_rate_hz=sampling_rate_hz,
                    K_global=K_global, drift_global=drift_global, lam_global=lam_global,
                    input_tag=input_tag, lag_global=lag_global,
                    null_tag=NULL_TAG,
                    null_percentile=RESPONDER_NULL_THRESH,
                    baseline_win_min=BASELINE_WIN_MIN,
                    drug_win_min=DRUG_WIN_MIN,
                    u_ext=u_ext,
                    overwrite=OVERWRITE_RESPONDERS,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
            delayed(_responders)(fish) for fish in tqdm(all_fish, desc="Responders")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    print("\n✅ GLM pipeline complete.")


if __name__ == "__main__":
    main()
