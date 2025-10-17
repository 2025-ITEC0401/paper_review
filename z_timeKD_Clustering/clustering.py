import os
import h5py
import cupy as cp
import cudf
from cuml.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

ROOT_DIR = './data'
DATASET = ['BasicMotions', 'Epilepsy', 'HandMovementDirection', 'Libras']
OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
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
        train_df = pd.DataFrame(train_data)
        
        with h5py.File(test_file, 'r') as f:
            test_data = f[KEY][:]
        test_df = pd.DataFrame(test_data)

        features = train_df.columns
        X_train = train_df[features]
        X_test = test_df[features]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        kmeans = KMeans(n_clusters=n_cluster, random_state=52, n_init=10)
        kmeans.fit(X_train_scaled)

        test_clusters = kmeans.predict(X_test_scaled)

        results_df = test_df.copy()
        results_df['cluster'] = test_clusters
        
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
        
        print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST))}) Target: {ds}_o{output_len}\n")
        idx += 1
        
        run_kmeans(h5_train_path, h5_test_path, f"{RES_DIR}/{ds}_o{output_len}_res.csv")

