import os
import h5py
import cupy as cp
import cudf
from cuml.manifold import TSNE
from cuml.decomposition import PCA
from cuml.cluster import DBSCAN

ROOT_DIR = './data'
DATASET = ['ETTh1', 'exchange_rate', 'traffic', 'electricity', 'HVAC']
OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
TYPE = ['train', 'val']
RES_DIR = './Result/csv'

os.makedirs(RES_DIR, exist_ok=True)

print("\n\n============= Clustering =============")

idx = 0
for ds in DATASET:
    for output_len in OUTPUT_LEN_LIST:
        for tp in TYPE:
            h5_path = f"{ROOT_DIR}/{ds}_o{output_len}_{tp}_consolidated.h5"
            
            if not os.path.exists(h5_path):
                print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST) * len(TYPE)}) File not found: {h5_path}")
                idx += 1
                continue
            
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST) * len(TYPE)}) Loading h5 file: {h5_path}\n")
            idx += 1

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
                method='barnes_hut',
                random_state=42
            )
            tsne_res_gpu = tsne_gpu.fit_transform(pca_result_gpu)
            print(f"t-SNE completed. Shape after t-SNE: {tsne_res_gpu.shape}")

            print("\nRunning DBSCAN on GPU...\n")
            dbscan_gpu = DBSCAN(eps=15, min_samples=10)
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
            
            output_filename = f'{RES_DIR}/{ds}_o{output_len}_{tp}_res.csv'
            df_plot_pd.to_csv(output_filename, index=False)
            print(f"Result Saved: {output_filename}\n")