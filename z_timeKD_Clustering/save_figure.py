import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import h5py
import pandas as pd
import cudf
from cuml.manifold import TSNE 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

DATASET = ['BasicMotions', 'Epilepsy', 'HandMovementDirection', 'Libras']
# OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
OUTPUT_LEN_LIST = [24]
TYPE = ['train', 'val']
CSV_DIR = './Result/csv'
RES_ROOT_DIR = './Result'
ROOT_DIR = './data'
KEY = 'embeddings'

print("\n\n============= Save Figure =============")

def concatenation(h5_path, n_vars):
    with h5py.File(h5_path, 'r') as f:
        data_unrolled = f[KEY][:]
        
    num_samples = len(data_unrolled) // n_vars
    embedding_dim = data_unrolled.shape[1]
    
    data_concatenated = data_unrolled.reshape(num_samples, n_vars * embedding_dim)
    
    return data_concatenated

idx = 1
for ds in DATASET:
    for output_len in OUTPUT_LEN_LIST:
        for method in ['kmeans', 'spectral']:
            csv_rawfile = f'{CSV_DIR}/{ds}_o{output_len}_{method}_res.csv'
            csv_file = f'{RES_ROOT_DIR}/{ds}_o{output_len}_{method}_tSNE_res.csv'
            fig_path = f'{RES_ROOT_DIR}/{ds}_o{output_len}_{method}_tSNE_res.png'
            h5_train_path = f"{ROOT_DIR}/{ds}_o{output_len}_{TYPE[0]}_consolidated.h5"
            h5_test_path = f"{ROOT_DIR}/{ds}_o{output_len}_{TYPE[1]}_consolidated.h5"
            
            if not os.path.exists(csv_rawfile) or not os.path.exists(h5_test_path) or not os.path.exists(h5_train_path):
                print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found.")
                idx += 1
                continue
            
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) Dataset: {ds}, Output: {output_len}... ", end="")
            idx += 1
            
            cluster_labels_df = pd.read_csv(csv_rawfile)
            
            match ds:
                case 'BasicMotions':
                    var = 6
                case 'Epilepsy':
                    var = 3
                case 'HandMovementDirection':
                    var = 10
                case 'Libras':
                    var = 2
            
            features_val_concat = concatenation(h5_test_path, n_vars=var)
            features_train_concat = concatenation(h5_train_path, n_vars=var)
            
            scaler = StandardScaler()
            scaler.fit(features_train_concat)
            features_val_scaled = scaler.transform(features_val_concat)

            reducer = TSNE(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(features_val_scaled)

            if hasattr(embedding_2d, 'get'):
                embedding_2d = embedding_2d.get()
                
            vis_df = pd.DataFrame(embedding_2d, columns=['tsne-1', 'tsne-2'])
            vis_df['cluster'] = cluster_labels_df['cluster']
            
            
            vis_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            plt.figure(figsize=(10, 8))
            n_clusters = len(vis_df['cluster'].unique())
            sns.scatterplot(
                x="tsne-1", y="tsne-2",
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