import h5py
import numpy as np
import cupy as cp
import cudf
from cuml.manifold import TSNE
from cuml.decomposition import PCA
from cuml.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = './data'
DATASET = ['ETTh1', 'exchange_rate', 'traffic', 'electricity', 'HVAC']
OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
TYPE = ['train', 'val']

for ds in DATASET:
    for output_len in OUTPUT_LEN_LIST:
        for tp in TYPE:
            h5_path = f"{ROOT_DIR}/{ds}_o{output_len}_{tp}_consolidated.h5"
            
            print("Loading h5 file...\n")
            with h5py.File(h5_path, 'r') as hf:
                loaded_matrix = hf['data'][:]
            
            print("Moving data from CPU to GPU...\n")
            gpu_embedding_vectors = cp.asarray(loaded_matrix)

            print("\nRunning PCA on GPU...\n")
            pca_gpu = PCA(n_components=50)
            pca_result_gpu = pca_gpu.fit_transform(gpu_embedding_vectors)
            
            print("\nRunning t-SNE on GPU...\n")
            tsne_gpu = TSNE(
                n_components=2,
                perplexity=30,
                method='barnes-hut',
                random_state=42
            )
            tsne_res_gpu = tsne_gpu.fit_transform(pca_result_gpu)
            print(f"t-SNE completed. Shape after t-SNE: {tsne_res_gpu.shape}")

            print("\nRunning DBSCAN on GPU...\n")
            dbscan_gpu = DBSCAN(eps=1.0, min_samples=5)
            clusters_gpu = dbscan_gpu.fit_predict(tsne_res_gpu)
            print("DBSCAN Completed.")
            
            print("\nMoving results back to CPU for plotting...")
            tsne_res_cpu = cp.asnumpy(tsne_res_gpu)
            cluster_cpu = cp.asnumpy(clusters_gpu)

            n_clusters = len(set(cluster_cpu)) - (1 if -1 in cluster_cpu else 0)
            print(f"Found {n_clusters} clusters.")
            
            df_plot = cudf.DataFrame()
            df_plot['tsne-2d-one'] = tsne_res_gpu[:, 0]
            df_plot['tsne-2d-two'] = tsne_res_gpu[:, 1]
            df_plot['cluster_label'] = clusters_gpu
            
            df_plot_pd = df_plot.to_pandas()
            
            output_filename = '{ROOT_DIR}/{ds}_o{output_len}_{tp}_res.csv'
            df_plot_pd.to_csv(output_filename, index=False)
            print("Result Saved.\n")
            
            # plt.figure(figsize=(14, 10))
            # sns.scatterplot(
            #     x="tsne-2d-one", y="tsne-2d-two",
            #     hue="cluster_label",
            #     palette=sns.color_palette("hsv", len(set(clusters_cpu))),
            #     data=df_plot_pd,
            #     legend="full",
            #     alpha=0.7
            # )
            
            # plt.title("GPU Accelerated t-SNE & DBScan Clustering")
            # plt.xlabel("t-SNE Dimension 1")
            # plt.ylabel("t-SNE Dimension 2")
            # plt.grid(True)
            # plt.show()