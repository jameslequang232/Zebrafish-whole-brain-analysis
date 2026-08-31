import sys
import pandas as pd
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

import os
import gc
from pathlib import Path
import numpy as np
import joblib
from scipy.stats import zscore
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ProcessPoolExecutor 
import ants
from sklearn.decomposition import FactorAnalysis

from concurrent.futures import ThreadPoolExecutor
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

BASE_DIR = Path("/mnt/storage-raid10/Yun/analysis_output/chemogenetic")
PROJ_ID = "hcrt-trpv1_huc-h2b-g8m_csn_120min"
PROJ_CTRL = "huc-h2b-g8m_csn_120min"
TEMPLATE_BRAIN_PATH = "/mnt/storage-raid10/Yun/analysis_output/registration/template_mean_brain.nii.gz"
SUMMARY_PLOTS_DIR = "/ssd-pool/james/lightsheet/Zebrafish-whole-brain-analysis/unsupervised_plots_summary"
stim_start_post = 2700
stim_end_post = 7200
PERCENTILE_CUTOFF = 90

EXPT_FISH_LIST = [
    #"251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4",
    #"251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    #"251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2",
    #"251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    #"251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2", 
    #"251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3",
    "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1",
    "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2",
    #"260515_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"
]



RESPONSE_TYPES = [
    "tonic_pos",
    "tonic_neg",
    "phasic_pos",
    "phasic_neg"
                 ]

DIR_ANTS_OUTPUT = str(BASE_DIR)


N_COMPONENTS = 15
N_CLUSTERS = 7

CATEGORY_COLORS = {'tonic_pos': 'orange', 'tonic_neg': 'blue', 'phasic_pos': 'red', 'phasic_neg': 'purple'} 
LABEL_TITLES = {'tonic_pos': 'Tonic(+)', 'tonic_neg': 'Tonic(-)', 'phasic_pos': 'Phasic(+)', 'phasic_neg': 'Phasic(-)'} 

from multiprocessing import Pool, cpu_count

def process_single_experiment(args):
    expt_ID, PERCENTILE_CUTOFF, PROJ_ID, DIR_ANTS_OUTPUT, medoid_types, stim_start, stim_end, use_qc, t, save_dendrogram, max_cells, n_components = args
    
    dir_expt = os.path.join(DIR_ANTS_OUTPUT, PROJ_ID, expt_ID)
    OUTPUT_DIR = os.path.join(dir_expt, f"FA_agglo_clustering_pct={PERCENTILE_CUTOFF}_c={t}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    loaded_traces = {}

    for label in medoid_types:
        is_tonic = "tonic" in label
        data_filename = "f_tonic.npy" if is_tonic else "f_phasic.npy"
        data_path = os.path.join(dir_expt, data_filename)
        
        if not os.path.exists(data_path):
            continue

        if data_filename not in loaded_traces:
            loaded_traces[data_filename] = np.load(data_path)
        traces = loaded_traces[data_filename]

        
        if is_tonic:
            tonic_score = np.var(traces[:, stim_start:stim_end], axis=1)
            
            if 'pos' in label:
                tonic_cutoff = np.percentile(tonic_score, PERCENTILE_CUTOFF)
                idxs = np.where(tonic_score >= tonic_cutoff)[0]
            else:
                tonic_cutoff = np.percentile(tonic_score, 100 - PERCENTILE_CUTOFF)
                idxs = np.where(tonic_score <= tonic_cutoff)[0]
            
        else:
            dprime_path = os.path.join(dir_expt, "phasic_dprime_cells_raw.npy")
            if not os.path.exists(dprime_path):
                continue
            dprime = np.load(dprime_path)
            
            
            if "pos" in label:
                top_percentile_cutoff = np.nanpercentile(dprime, PERCENTILE_CUTOFF)
          
                idxs = np.where(dprime >= top_percentile_cutoff)[0]
            else:
                top_percentile_cutoff = np.nanpercentile(dprime, 100 - PERCENTILE_CUTOFF)
        
                idxs = np.where(dprime <= top_percentile_cutoff)[0]

          
            print(f"\n[{label} COMPUTE CHECK]")
            print(f"Top percentile value calculated: {top_percentile_cutoff}")
            print(f"Extracted array length before check: {len(idxs)}")
            
            idxs = idxs[idxs < traces.shape[0]]

            print(f" [{label}] Cells after bounds check: {len(idxs)}")

        if len(idxs) == 0:
            print(f" Warning: [{expt_ID} - {label}] No cells passed the percentile threshold.")
            continue
            
        trace_subset = traces[idxs, stim_start:stim_end]

        row_variances = np.var(trace_subset, axis=1)
        nonzero_variance_mask = row_variances > 1e-8
        
        if np.sum(nonzero_variance_mask) < 4:
            print(f" Warning: [{expt_ID} - {label}] Dropped. Only {np.sum(nonzero_variance_mask)} cells had non-zero variance during the stim window.")
            continue
            
      
        trace_subset = trace_subset[nonzero_variance_mask]
        idxs = idxs[nonzero_variance_mask]
        
        z_traces = zscore(trace_subset, axis=1)
        valid_mask = np.all(np.isfinite(z_traces), axis=1)
        
        if np.sum(valid_mask) < 4:
            print(f" Warning: [{expt_ID} - {label}] Dropped. Z-scoring still resulted in non-finite values.")
            continue
            
        z_traces = z_traces[valid_mask]
        idxs = idxs[valid_mask]

        MAX_TARGET_CELLS = 50000
        if len(idxs) > MAX_TARGET_CELLS:
    
            metric_arr = tonic_score[idxs] if is_tonic else dprime[idxs]
            
            if "pos" in label:
                top_orders = np.argsort(metric_arr)[-MAX_TARGET_CELLS:]
            else:
                top_orders = np.argsort(metric_arr)[:MAX_TARGET_CELLS]
        
            z_traces = z_traces[top_orders]
            idxs = idxs[top_orders]



        print(f"[{expt_ID} - {label}] Processing FA + Clustering with {z_traces.shape[0]} cells...")

        try:
            fa = FactorAnalysis(n_components=n_components, random_state=0)
            latent = fa.fit_transform(z_traces)
            fa_model_path = os.path.join(OUTPUT_DIR, f"FA_model_{label}.joblib")
            joblib.dump(fa, fa_model_path)

            clusterer = AgglomerativeClustering(n_clusters=t, linkage='ward')
            cluster_labels = clusterer.fit_predict(latent)

            for c_idx in range(1, t + 1):

                c_mask = (cluster_labels == (c_idx - 1))
                cluster_cell_ids = idxs[c_mask]
                out_idx_path = os.path.join(OUTPUT_DIR, f"{label}_c{c_idx}_idxs.npy")
                np.save(out_idx_path, cluster_cell_ids)

            del z_traces, latent, trace_subset, idxs
        except Exception as e:
            print(f"Failure processing cluster execution line for {label}: {e}")
            continue
        
    del loaded_traces
    gc.collect()
    return expt_ID

def hierarchical_clustering_parallel(
    EXPT_FISH_LIST, PERCENTILE_CUTOFF, PROJ_ID, DIR_ANTS_OUTPUT, RESPONSE_TYPES, 
    stim_start, stim_end, use_qc, t, save_dendrogram, max_cells, n_components
):
    tasks = [(
        expt_ID, PERCENTILE_CUTOFF, PROJ_ID, DIR_ANTS_OUTPUT, RESPONSE_TYPES, 
        stim_start, stim_end, use_qc, t, save_dendrogram, max_cells, n_components
    ) for expt_ID in EXPT_FISH_LIST]
    
    num_workers = min(4, max(1, int(cpu_count() * 0.75)))
    print(f"Starting parallel execution with {num_workers} workers.")
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_experiment, tasks), total=len(tasks), desc="Parallel clustering"))        
    return results

if __name__ == "__main__":
    print("STEP 1: Running Parallel Agglomerative Clustering")

    hierarchical_clustering_parallel(
        EXPT_FISH_LIST=EXPT_FISH_LIST, PERCENTILE_CUTOFF=PERCENTILE_CUTOFF, PROJ_ID=PROJ_ID,
        DIR_ANTS_OUTPUT=DIR_ANTS_OUTPUT, RESPONSE_TYPES=RESPONSE_TYPES,
        stim_start=2700, stim_end=7200, use_qc = False,  t=N_CLUSTERS, n_components=N_COMPONENTS, 
        save_dendrogram=False,
        max_cells=50000
    )

loaded_cluster_data = {}
def process_fish_analysis(fish_id):
    fish_dir = BASE_DIR / PROJ_ID / fish_id

    cluster_dir = fish_dir / f'FA_agglo_clustering_pct={PERCENTILE_CUTOFF}_c={N_CLUSTERS}'
        
    if not cluster_dir.exists():
        print(f"Skipping {fish_id} — Clustering output directory not found at {cluster_dir}")
        return None
    vox_path = fish_dir / "medoids_template_vox.npy"
    if not vox_path.exists():
        print(f"Skipping {fish_id} — Coordinates array missing.")
        return None
    medoids_vox = np.load(vox_path)
    
    fish_results = {cat: {} for cat in RESPONSE_TYPES}
    
    for category in RESPONSE_TYPES:
        is_tonic = "tonic" in category
        data_filename = "f_tonic.npy" if is_tonic else "f_phasic.npy"
        data_path = fish_dir / data_filename
        
        if not data_path.exists():
            print(f'{data_path} does not exist')
            continue
            
        try:
            all_cluster_cells = []
            cluster_cell_mappings = {}
            
            for c_idx in range(1, N_CLUSTERS + 1):
                idx_file = cluster_dir / f"{category}_c{c_idx}_idxs.npy"
                if idx_file.exists():
                    c_cells = np.load(idx_file)
                    if len(c_cells) > 0:
                        all_cluster_cells.append(c_cells)
                        cluster_cell_mappings[c_idx] = c_cells
                        
            if not all_cluster_cells:
                continue
                
            unique_responders = np.unique(np.concatenate(all_cluster_cells))
                
            raw_traces_mmap = np.load(data_path, mmap_mode="r")
            pooled_post_traces = np.array(raw_traces_mmap[unique_responders, stim_start_post:stim_end_post])
            pooled_full_traces = np.array(raw_traces_mmap[unique_responders, 0:7200])
            cell_to_ram_idx = {cell: i for i, cell in enumerate(unique_responders)}
            
            for c_idx in range(1, N_CLUSTERS + 1):
                if c_idx not in cluster_cell_mappings:
                    continue
                    
                cluster_cells = cluster_cell_mappings[c_idx]
                ram_indices = [cell_to_ram_idx[cell] for cell in cluster_cells]
                
                c_post = pooled_post_traces[ram_indices]
                c_full = pooled_full_traces[ram_indices]
                
                z_post = zscore(c_post, axis=1)
                z_full = zscore(c_full, axis=1)
                
                valid_mask = np.all(np.isfinite(z_post), axis=1) & np.all(np.isfinite(z_full), axis=1)
                
                if np.sum(valid_mask) > 0:
            
                    final_cluster_cells = cluster_cells[valid_mask]
                    
                    fish_results[category][c_idx] = {
                        'post_drug': z_post[valid_mask],
                        'full_timeline': z_full[valid_mask],
                        'cell_ids': final_cluster_cells,
                        'coords': medoids_vox[final_cluster_cells] 
                    }
                    
        except Exception as e:
            print(f"Error loading {category} for {fish_id}: {e}")
            import traceback 
            print(f"\nCRITICAL CRASH inside {category} for fish {expt_ID}:")
            print(f"Error Type: {str(e)}")
            traceback.print_exc()
            continue
            
    return fish_id, fish_results


with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(process_fish_analysis, EXPT_FISH_LIST)
    for result in results:
        if result is not None:
            fish_id, fish_results = result
            loaded_cluster_data[fish_id] = fish_results

print("Data loading completed successfully")


# USE THIS
if not os.path.exists(TEMPLATE_BRAIN_PATH): 
    raise FileNotFoundError(f"Missing mandatory anatomical canvas template: {TEMPLATE_BRAIN_PATH}") 

nii_obj = nib.load(TEMPLATE_BRAIN_PATH) 
template_img = nii_obj.get_fdata() 

bg_canvas = np.max(template_img, axis=2).T 

plt.ioff() 

for expt_ID in EXPT_FISH_LIST: 
    print(f"\nEvaluating raw responder maps for: {expt_ID}") 
    fish_dir = BASE_DIR / PROJ_ID / expt_ID 
    vox_path = fish_dir / "medoids_template_vox.npy" 
    
    if not vox_path.exists(): 
        print(f" Warning: {vox_path} missing. Skipping this fish.") 
        continue 
        
    medoids_vox = np.load(vox_path) 
    valid_registration_mask = (medoids_vox >= 0).all(axis=1) 
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14)) 
    plt.suptitle(f"Pre-Clustering Whole Responder Distributions\n{expt_ID} (Top {100-PERCENTILE_CUTOFF}% Cutoff)", fontsize=15, y=0.98)
    axes = axes.flatten() 
    
    for idx, category in enumerate(RESPONSE_TYPES): 
        ax = axes[idx] 
        

        cluster_dir = fish_dir / f'FA_agglo_clustering_pct={PERCENTILE_CUTOFF}_c={N_CLUSTERS}'
        pipeline_idxs = [] 
        for c_idx in range(1, N_CLUSTERS + 1): 
            idx_file = cluster_dir / f"{category}_c{c_idx}_idxs.npy" 
            if idx_file.exists(): 
                pipeline_idxs.append(np.load(idx_file)) 
                
        if len(pipeline_idxs) == 0: 
            ax.text(0.5, 0.5, f"Missing cluster files for:\n{category}", ha='center', va='center', color='gray') 
            ax.axis("off") 
            continue 
            
        raw_responder_idxs = np.concatenate(pipeline_idxs) 
        valid_raw_idxs = raw_responder_idxs[valid_registration_mask[raw_responder_idxs]] 
        total_cells_found = len(valid_raw_idxs) 
        
        ax.imshow(bg_canvas, cmap="gray", origin="upper") 
        
        if total_cells_found == 0: 
            ax.text(0.5, 0.5, f"No Responding Cells\n(n = 0)", ha='center', va='center', color='gray') 
            ax.set_title(LABEL_TITLES[category], fontsize=12) 
            ax.axis("off") 
            continue 
            
        coords = medoids_vox[valid_raw_idxs] 
    
        ax.scatter( 
            coords[:, 0], 
            coords[:, 1], 
            color=CATEGORY_COLORS[category], 
            s=1.2, 
            alpha=0.35, 
            edgecolors='none' 
        ) 
        
        ax.set_title(f"{LABEL_TITLES[category]} Responder Pool (n = {total_cells_found} cells)", fontsize=13) 
        ax.set_xlim(0, bg_canvas.shape[1]) 
     
        ax.set_ylim(bg_canvas.shape[0], 0) 
        ax.axis("off") 
        
    fig.subplots_adjust(wspace=0.3, hspace=0.3) 
    fish_summary_dir = os.path.join(SUMMARY_PLOTS_DIR, expt_ID) 
    os.makedirs(fish_summary_dir, exist_ok=True) 
    
    output_path = os.path.join(fish_summary_dir, f"whole_distribution_plot_pct_{PERCENTILE_CUTOFF}.png")
    fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.4) 
    print(f" Successfully saved unified raw pool grid -> {output_path}") 
    plt.close(fig) 
    gc.collect() 

plt.ion()

#this one - plot spatila dsteibution into clusters
TEMPLATE_BRAIN_PATH = "/mnt/storage-raid10/Yun/analysis_output/registration/template_mean_brain.nii.gz"
SUMMARY_PLOTS_DIR = "/ssd-pool/james/lightsheet/Zebrafish-whole-brain-analysis/unsupervised_plots_summary"
COLOR_PALETTE = plt.cm.get_cmap('tab10', N_CLUSTERS)

if not os.path.exists(TEMPLATE_BRAIN_PATH):
    raise FileNotFoundError(f"Missing mandatory anatomical canvas template: {TEMPLATE_BRAIN_PATH}")

nii_obj = nib.load(TEMPLATE_BRAIN_PATH)
template_img = nii_obj.get_fdata()

bg_canvas = np.max(template_img, axis=2).T 
plt.ioff()

for expt_ID in EXPT_FISH_LIST:
    if expt_ID not in loaded_cluster_data:
        continue
        
    print(f"\nProjecting anatomical spatial profiles for: {expt_ID}")
    fish_dir = BASE_DIR / PROJ_ID / expt_ID
    
    vox_path = fish_dir / "medoids_template_vox.npy"
    if not vox_path.exists():
        print(f"  Missing template voxels for {expt_ID}. Skipping.")
        continue
        
    medoids_vox = np.load(vox_path)
    valid_registration_mask = (medoids_vox >= 0).all(axis=1)

    cluster_dir = fish_dir / f'FA_agglo_clustering_pct={PERCENTILE_CUTOFF}_c={N_CLUSTERS}'
    
    for category in RESPONSE_TYPES:
        target_subfolder = os.path.join(SUMMARY_PLOTS_DIR, expt_ID, category)
        os.makedirs(target_subfolder, exist_ok=True)
        
        valid_cluster_files = []
        for c_idx in range(1, N_CLUSTERS + 1):
            idx_file = cluster_dir / f"{category}_c{c_idx}_idxs.npy"
            if idx_file.exists():
                cluster_indices = np.load(idx_file)
                if len(cluster_indices) > 0:
                    valid_cluster_indices = cluster_indices[valid_registration_mask[cluster_indices]]
                    if len(valid_cluster_indices) > 0:
                        valid_cluster_files.append((c_idx, idx_file, valid_cluster_indices))
                        
        n_valid_panels = len(valid_cluster_files)
        if n_valid_panels == 0:
            continue
     
        n_cols = min(6, n_valid_panels)
        n_rows = int(np.ceil(n_valid_panels / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
        plt.suptitle(f"{expt_ID} - {category.upper()} Spatial Cluster Maps (Top {100-PERCENTILE_CUTOFF}% Cutoff)", fontsize=15, y=0.98)
        axes = axes.flatten()
        
        plotted_panels = 0
        
        for c_idx, idx_file, valid_cluster_indices in valid_cluster_files:
            if plotted_panels >= len(axes):
                print(f"Omitted extra cluster panel {c_idx} to avoid array out-of-bounds")
                break
                
            coords = medoids_vox[valid_cluster_indices]
            ax = axes[plotted_panels]
  
         
            ax.imshow(bg_canvas, cmap="gray", origin="upper")
 
            ax.scatter(
                coords[:, 0], 
                coords[:, 1], 
                color=COLOR_PALETTE(c_idx - 1),
                s=2.5, 
                alpha=0.6, 
                edgecolors='none'
            )
            
            ax.set_title(f"Cluster {c_idx} (n={len(valid_cluster_indices)} cells)", fontsize=11)
            ax.set_xlim(0, bg_canvas.shape[1])

            ax.set_ylim(bg_canvas.shape[0], 0) 
            ax.axis("off")
            
            ax.text(0.05, 0.08, f"C{c_idx}", 
                    transform=ax.transAxes, 
                    color="white", 
                    fontsize=20, 
                    weight="bold", 
                    ha="left", 
                    va="bottom")
            plotted_panels += 1
            
        for empty_idx in range(plotted_panels, len(axes)):
            fig.delaxes(axes[empty_idx])
            
        if plotted_panels > 0:
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            output_path = os.path.join(target_subfolder, f"spatial_distribution_plot_pct_{PERCENTILE_CUTOFF}.png")
            fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.5)
            print(f"Saved plot for: {output_path}")
            
        plt.close(fig)
        
    gc.collect()

plt.ion()
print("\nAll corrected plots rendered successfully.")