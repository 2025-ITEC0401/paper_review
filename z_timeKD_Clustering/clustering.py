import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import h5py
import cudf
import cupy as cp
from cuml.cluster import KMeans
import pandas as pd
import numpy as np

ROOT_DIR = './data'
# DATASET = ['BasicMotions', 'Epilepsy', 'HandMovementDirection', 'Libras']
# OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
DATASET = ['BasicMotions', 'Epilepsy', 'Libras']
OUTPUT_LEN_LIST = [24]
TYPE = ['train', 'val']
RES_DIR = './Result/csv'
KEY = 'embeddings'

os.makedirs(RES_DIR, exist_ok=True)

def run_kmeans(train_file, test_file, output_file):
    try:
        match ds:
            case 'BasicMotions':
                n_cluster = 4
            case 'Epilepsy':
                n_cluster = 4
            case 'HandMovementDirection':
                n_cluster = 4
            case 'Libras':
                n_cluster = 15
                
        with h5py.File(train_file, 'r') as f:
            train_data = f[KEY][:]
        train_gdf = cudf.DataFrame(train_data)
        
        with h5py.File(test_file, 'r') as f:
            test_data = f[KEY][:]
        test_gdf = cudf.DataFrame(test_data)

        kmeans_gpu = KMeans(n_clusters=n_cluster, random_state=52)        
        kmeans_gpu.fit(train_gdf)

        test_clusters = kmeans_gpu.predict(test_gdf)

        results_df = test_gdf.to_pandas()
        results_df['cluster'] = test_clusters.to_numpy()
        
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
        
        run_kmeans(h5_train_path, h5_test_path, f"{RES_DIR}/{ds}_o{output_len}_res.csv")

