import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import cudf
from cuml.manifold import TSNE 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# DATASET = ['BasicMotions', 'Epilepsy', 'HandMovementDirection', 'Libras']
# OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
DATASET = ['BasicMotions', 'Epilepsy', 'Libras']
OUTPUT_LEN_LIST = [24]
TYPE = ['train', 'val']
CSV_DIR = './Result/csv'
RES_ROOT_DIR = './Result'

print("\n\n============= Save Figure =============")

idx = 1
for ds in DATASET:
    for output_len in OUTPUT_LEN_LIST:
        csv_rawfile = f'{CSV_DIR}/{ds}_o{output_len}_res.csv'
        csv_file = f'{RES_ROOT_DIR}/{ds}_o{output_len}_tSNE_res.csv'
        fig_path = f'{RES_ROOT_DIR}/{ds}_o{output_len}_tSNE_res.png'
        
        if not os.path.exists(csv_rawfile):
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found: {csv_rawfile}")
            idx += 1
            continue
        
        print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File: {fig_path}... ", end="")
        idx += 1
        
        results_df = pd.read_csv(csv_rawfile)
        
        if 'cluster' not in results_df.columns:
            print("ERROR: Cannot find 'cluster' column.")
            continue       
        
        features_cpu = results_df.drop('cluster', axis=1)
        cluster_labels_cpu = results_df['cluster']
        
        scaler = StandardScaler()
        features_scaled_cpu = scaler.fit_transform(features_cpu)
        features_scaled_gpu = cudf.DataFrame(features_scaled_cpu)
        
        tsne_gpu = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results_gpu = tsne_gpu.fit_transform(features_scaled_gpu)
        
        vis_df = tsne_results_gpu.to_pandas()
        vis_df.columns = ['tsne-2d-one', 'tsne-2d-two']
        
        tsne_results_cpu = tsne_results_gpu.to_pandas()
        vis_df['cluster'] = cluster_labels_cpu.values
        
        vis_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        plt.figure(figsize=(10, 8))
        n_clusters = len(vis_df['cluster'].unique())
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="cluster",
            palette=sns.color_palette("hsv", n_clusters),
            data=vis_df,
            legend="full",
            alpha=0.8
        )

        plt.title(f"t-SNE visualization\nDataset: {ds}, Output Length: {output_len}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True)

        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        print("Completed.")

        plt.close()