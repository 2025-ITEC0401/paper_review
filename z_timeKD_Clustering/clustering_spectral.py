import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import h5py
import cudf
import cupy as cp
import cugraph
from cuml.preprocessing import StandardScaler
from cuml.manifold import UMAP
from cuml.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = './data'
DATASET = ['BasicMotions', 'Epilepsy', 'HandMovementDirection', 'Libras']
# OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
OUTPUT_LEN_LIST = [24]
TYPE = ['train', 'val']
RES_DIR = './Result/csv'
KEY = 'embeddings'
N_NEIGHBORS = 15

os.makedirs(RES_DIR, exist_ok=True)

def visualize(scaled_data_gdf, title, output_path):
    scaled_data_np = scaled_data_gdf.to_numpy()
    reducer = UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(scaled_data_np)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        alpha=0.7
    )
    plt.title(f'UMAP Projection of {title}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UAP Dimension 2')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def concatenation(h5_path, n_vars):
    with h5py.File(h5_path, 'r') as f:
        data_unrolled = f[KEY][:]
        
    num_samples = len(data_unrolled) // n_vars
    embedding_dim = data_unrolled.shape[1]
    
    data_concatenated = data_unrolled.reshape(num_samples, n_vars * embedding_dim)
    
    return data_concatenated

def run_kmeans(ds, train_file, test_file, output_file):
    try:
        match ds:
            case 'BasicMotions':
                var = 6
            case 'Epilepsy':
                var = 3
            case 'HandMovementDirection':
                var = 10
            case 'Libras':
                var = 2
                
        match ds:
            case 'BasicMotions':
                n_cluster = 4
            case 'Epilepsy':
                n_cluster = 4
            case 'HandMovementDirection':
                n_cluster = 4
            case 'Libras':
                n_cluster = 15
        
        train_data_concat = concatenation(train_file, n_vars=var)
        val_data_concat = concatenation(test_file, n_vars=var)
        
        train_gdf_concat = cudf.DataFrame(train_data_concat)
        val_gdf_concat = cudf.DataFrame(val_data_concat)
        
        scaler = StandardScaler()
        scaler.fit(train_gdf_concat)
        val_gdf_scaled = scaler.transform(val_gdf_concat)
        
        # visualize(val_gdf_scaled, f"{ds} Val data", f"./{ds}_val_gdf.png")
        
        knn_model = NearestNeighbors(n_neighbors=N_NEIGHBORS)
        knn_model.fit(val_gdf_scaled)
        
        indices = knn_model.kneighbors(val_gdf_scaled, return_distance=False)

        n_samples = val_gdf_scaled.shape[0]
        
        source = cp.repeat(cp.arange(n_samples, dtype='int32'), N_NEIGHBORS)

        destination = indices.to_cupy().flatten().astype('int32')
        
        edges_df = cudf.DataFrame({
            'source': source,
            'destination': destination
        })
        
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges_df, source='source', destination='destination', renumber=False)

        result_df = cugraph.spectralBalancedCutClustering(
            G,
            num_clusters=n_cluster,
            num_eigen_vects=n_cluster
        )
        
        result_df = result_df.sort_values(by='vertex')        
        predicted_lables_final = result_df['cluster'].to_numpy()

        results_df = pd.DataFrame({'cluster': predicted_lables_final})
        
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"Result Saved: {output_file}\n")
    except KeyError:
        print(f"Cannot find key in h5 file.\n")
    except Exception as e:
        print(f"ERROR: {e}")

print("\n\n============= Clustering =============")

idx = 1
for ds in DATASET:
    for output_len in OUTPUT_LEN_LIST:
        h5_train_path = f"{ROOT_DIR}/{ds}_o{output_len}_{TYPE[0]}_consolidated.h5"
        h5_test_path = f"{ROOT_DIR}/{ds}_o{output_len}_{TYPE[1]}_consolidated.h5"
        
        if not os.path.exists(h5_train_path):
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found: {h5_train_path}")
            idx += 1
            continue
        
        if not os.path.exists(h5_test_path):
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found: {h5_test_path}")
            idx += 1
            continue
        
        print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) Target: {ds}_o{output_len}\n")
        idx += 1
        
        run_kmeans(ds, h5_train_path, h5_test_path, f"{RES_DIR}/{ds}_o{output_len}_spectral_res.csv")

