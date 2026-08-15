import os
import sys
import gc
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import nibabel as nib
import ants
import numpy as np

from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import TruncatedSVD
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import importlib
from collections import defaultdict
import traceback
from matplotlib.backends.backend_pdf import PdfPages
from typing import Union
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

from scipy.ndimage import percentile_filter, gaussian_filter
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import fftconvolve
import pickle
import tifffile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


BASE_DIR = Path("/mnt/storage-raid10/Yun/analysis_output/chemogenetic")
PROJ_ID = "hcrt-trpv1_huc-h2b-g8m_csn_120min"
PROJ_CTRL = "huc-h2b-g8m_csn_120min"
EXPT_FISH_LIST = [
    "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4",
    "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2",
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2", 
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3",
    "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2",
    "260515_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"
]

N_COMPONENTS = 5
N_CLUSTERS = 7
PHASIC_DPRIME_THRESH = 0.25
RESPONSE_TYPES = ["tonic_pos", "tonic_neg", "phasic_pos", "phasic_neg"]

DIR_ANTS_OUTPUT = str(BASE_DIR)

example_fish = '251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4' #used for testing

#Parameters 

# IMAGING SPECS 
sec_per_volume = 1
volume_per_sec = 1
n_slices = 40
depth = 250
binning = 1
res_x = 1.52*binning
res_y = 1.52*binning
res_z = depth/n_slices
rotation_k = 2 

# DRUG PERFUSION PARAMETER 
drug_uM   = 10.0
V_ml     = 15.0
Q_ml_min = 4.5


baseline_start = 0 * 60 * volume_per_sec
baseline_end = 45 * 60 * volume_per_sec

# define drug perfusion start & end time frame 
drug_start = 46  * 60 * volume_per_sec
drug_end = 90  * 60 * volume_per_sec

# define E3+DMSO wash out start & end time frame
wash_start = 91 * 60 * volume_per_sec
wash_end = 120 * 60 * volume_per_sec


#  define delta F / F 's baseline percentile
df_f_percentile = 20

# define F_tonic's window size and percentile
f_tonic_window_size = 600  #seconds
f_tonic_percentile = 20

# permutation test parameters
p_thresh_permutation = 0.005
n_resample_permutation = 500

# RUN BH-FDR FILTERING CODE
BH_Q = 0.05  



input_tag    = "C"
K_global     = 600
drift_global = 1   
lam_global   = 0.5
lag_global   = 0  

param_folder_name = f"in{input_tag}_K{K_global}_drift{drift_global}_lam{lam_global}_lag{lag_global}"

CLIP_ABS_DZ = 50.0 
INCLUDED_BASELINE = 15.0 
NULL_TAG = "iaaft"
RESPONDER_NULL_THRESH = 95 
L_MIN = 20.0  
BASE_DIR = Path("/mnt/storage-raid10/Yun/analysis_output/chemogenetic")
PROJ_ID = "hcrt-trpv1_huc-h2b-g8m_csn_120min"

EXPT_FISH_LIST = [
    "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4",
    "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2",
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2",
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3",
    "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2",
    "260515_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"
]

N_COMPONENTS = 15
N_CLUSTERS = 7
PHASIC_DPRIME_THRESH = 0.25 
RESPONSE_TYPES = ["tonic_pos", "tonic_neg", "phasic_pos", "phasic_neg"]

def process_single_experiment(args):
    expt_ID, stim_start, stim_end, max_cells, n_components, n_clusters = args
    print(f"\nProcessing Group: {expt_ID}")
    
    dir_expt = BASE_DIR / PROJ_ID / expt_ID
 
    out_dir = dir_expt / f"FA_agglo_clustering_f={n_components}_c={n_clusters}_t={n_clusters}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for label in RESPONSE_TYPES:
        is_tonic = "tonic" in label
        data_filename = "f_tonic.npy" if is_tonic else "f_phasic.npy"
        data_path = dir_expt / data_filename
        
        if not data_path.exists():
            continue
            
        try:
            traces = np.load(data_path)
            
    
            if is_tonic:
                idx_filename = f"{label}_glm_iaaft_nullp95_idxs.npy"
                idx_path = dir_expt / idx_filename
                if not idx_path.exists():
                    continue
                idxs = np.load(idx_path)
            else:
                dprime_path = dir_expt / "phasic_dprime_cells_raw.npy"
                if not dprime_path.exists():
                    continue
                dprime = np.load(dprime_path)
                if "pos" in label:
                    idxs = np.where(dprime >= PHASIC_DPRIME_THRESH)[0]
                else:
                    idxs = np.where(dprime <= -PHASIC_DPRIME_THRESH)[0]
                    
            idxs = idxs[idxs < traces.shape[0]]
            if len(idxs) == 0:
                continue
                
    
            trace_subset = traces[idxs, stim_start:stim_end]
            z_traces = zscore(trace_subset, axis=1)
            
            valid_mask = np.all(np.isfinite(z_traces), axis=1)
            if np.sum(valid_mask) < n_clusters:
                continue
                
            z_traces = z_traces[valid_mask]
            idxs = idxs[valid_mask]

            current_pool_size = z_traces.shape[0]
            actual_sample_size = min(max_cells, current_pool_size)
            
            if current_pool_size > max_cells:
                selected_idx = np.random.choice(current_pool_size, actual_sample_size, replace=False)
                z_traces = z_traces[selected_idx]
                idxs = idxs[selected_idx]

            fa = FactorAnalysis(n_components=n_components, random_state=0)
            latent = fa.fit_transform(z_traces)
            
    
            joblib.dump(fa, out_dir / f"FA_model_{label}.joblib")
            
    
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clustering.fit_predict(latent) + 1
            
    
            total_saved = 0
            for i in range(n_clusters):
                cluster_indices = idxs[cluster_labels == (i + 1)]
                out_path = out_dir / f"{label}_c{i+1}_idxs.npy"
                np.save(out_path, cluster_indices)
                total_saved += len(cluster_indices)
                
            print(f"{label}: Successfully clustered and saved {total_saved} cells.")
            
            del z_traces, latent, trace_subset, idxs
            gc.collect()
            
        except Exception as e:
            print(f"Failed compiling processing layer {label} inside {expt_ID}: {e}")
            continue

if __name__ == "__main__":
    print("Launching production clustering run with dynamic sampling safeguards...")
    for fish in EXPT_FISH_LIST:
        process_single_experiment((fish, 2700, 7200, 5000, N_COMPONENTS, N_CLUSTERS))
    print("\nAll directories populated successfully. Files are ready for loading!")
