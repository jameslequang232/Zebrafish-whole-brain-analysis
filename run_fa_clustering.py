
import os
import gc
from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.backends.backend_pdf import PdfPages

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
PHASIC_DPRIME_THRESH = 0.5
RESPONSE_TYPES = ["tonic_pos", "tonic_neg", "phasic_pos", "phasic_neg"]

print("Step one of the clustering notebook (FA + Agglomerative Hierarchical Clustering)")


for fish_id in EXPT_FISH_LIST:
    fish_path = BASE_DIR / PROJ_ID / fish_id
    out_dir = fish_path / f"FA_agglo_clustering_results_POST_DRUG"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing Directory: {fish_id}")
    print(out_dir.resolve())

    for category in RESPONSE_TYPES:
        try:
            if "tonic" in category:
                trace_file = fish_path / "f_tonic.npy"
                if not trace_file.exists(): continue
                traces = np.load(trace_file)
                
                idx_file = fish_path / f"{category}_glm_iaaft_nullp99_idxs.npy"
                if not idx_file.exists(): continue
                cell_idxs = np.load(idx_file)
                selected_traces = traces[cell_idxs]
            else:
                trace_file = fish_path / "f_phasic.npy"
                if not trace_file.exists(): continue
                traces = np.load(trace_file)
                
                dprime_file = fish_path / "phasic_dprime_cells_raw.npy"
                if not dprime_file.exists(): continue
                dprime = np.load(dprime_file)
                
                if category == "phasic_pos":
                    cell_idxs = np.where(dprime >= PHASIC_DPRIME_THRESH)[0]
                else:
                    cell_idxs = np.where(dprime <= -PHASIC_DPRIME_THRESH)[0]
                selected_traces = traces[cell_idxs]

            if len(cell_idxs) < N_COMPONENTS:
                print(f" Skipped {category} because cell count is too low.")
                continue

            post_drug_traces = selected_traces[:, 2700:7200] #updated this to focus on post-drug time window (2700-7200) for pipeline
            valid_mask = np.std(post_drug_traces, axis=1) > 0
            clean_traces = post_drug_traces[valid_mask]
            clean_idxs = cell_idxs[valid_mask]
            
            z_traces = zscore(clean_traces, axis=1)
            print(f"{category}: Clustering {z_traces.shape[0]} cells across timeline.")


            fa = FactorAnalysis(n_components=N_COMPONENTS, random_state=0)
            latent_space = fa.fit_transform(z_traces) 
            

            joblib.dump(fa, out_dir / f"FA_model_{category}.joblib")

            factor_variance = np.var(latent_space, axis=0)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(1, N_COMPONENTS + 1), factor_variance, marker='o', color='purple')
            ax.set_title(f"Factor Variance — {category} — {fish_id}")
            ax.set_xlabel("Factor Index")
            ax.set_ylabel("Variance Value")
            fig.tight_layout()
            plt.savefig(out_dir / f"factors_variance_{category}.png", dpi=150)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(N_COMPONENTS):
                ax.plot(fa.components_[i], label=f"F{i+1}", alpha=0.7)
            ax.set_title(f"Factor Time Traces — {category} — {fish_id}")
            ax.set_xlabel("Timepoint Volumes")
            ax.set_ylabel("Component Weight Loading")
            ax.legend(loc='upper right', fontsize=6, ncol=5)
            fig.tight_layout()
            plt.savefig(out_dir / f"factor_time_traces_{category}.png", dpi=200)
            plt.close()

            pdf_path = out_dir / f"top10cells_allfactors_{category}.pdf"
            with PdfPages(pdf_path) as pdf:
                for i in range(min(5, N_COMPONENTS)):
                    factor_trace = fa.components_[i]
                    correlations = np.array([
                        np.corrcoef(z_traces[j], factor_trace)[0, 1]
                        if np.all(np.isfinite(z_traces[j])) else -np.inf
                        for j in range(z_traces.shape[0])
                    ])
                    top_cell_picks = np.argsort(correlations)[-10:]
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for j in top_cell_picks:
                        ax.plot(z_traces[j], alpha=0.7)
                    ax.set_title(f"Top 10 Cells — Factor {i+1} — {category}")
                    ax.set_xlabel("Timepoint")
                    ax.set_ylabel("Z-score Signal")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            agglo = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
            cluster_labels = agglo.fit_predict(latent_space) + 1 
            
            np.save(out_dir / f"cluster_labels_{category}.npy", cluster_labels)
            np.save(out_dir / f"selected_cell_idxs_{category}.npy", clean_idxs)


            if latent_space.shape[0] <= 30000:
                plot_data = latent_space.copy()
                if plot_data.shape[0] > 5000:
                    np.random.seed(0)  # For reproducibility
                    sampled_picks = np.random.choice(plot_data.shape[0], 5000, replace=False)
                    plot_data = plot_data[sampled_picks]
                    
                linkage_tree = linkage(plot_data, method='ward')
                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(linkage_tree, no_labels=True, ax=ax)
                ax.set_title(f"Hierarchical Clustering Dendrogram — {fish_id} — {category}")
                fig.tight_layout()
                plt.savefig(out_dir / f"dendrogram_{category}.png", dpi=200)
                plt.close()

            print(f" {category}: Partitioned cells cleanly. Plots saved to disk.")
            
        except Exception as e:
            print(f" Error executing pipeline category {category}: {e}")
            
    gc.collect()


print("finished fa and clustering for all fish")