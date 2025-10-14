import h5py
import numpy as np
import os
from glob import glob

EMBEDDING_KEY = 'embeddings'

for dataset in ['ETTh1', 'exchange_rate', 'traffic', 'electricity', 'HVAC']:
    for output_len in [24, 36, 48, 96, 192]:
        h5_dir_list = [f'./data/{dataset}/{output_len}/train', f'./data/{dataset}/{output_len}/val']
        for h5_dir in h5_dir_list:
            h5_files = glob(os.path.join(h5_dir, '*.h5'))
            if not h5_files:
                print(f"h5 file cannot be found. Check the path. {h5_dir}")
                continue
            print(f"\nDataset Name: {dataset}  |  Output Length: {output_len}  |  Path: {h5_dir}")
            print(f"The number of h5 Files: {len(h5_files)}")

            all_embeddings = []
            processed_count = 0
            for path in h5_files:
                try:
                    with h5py.File(path, 'r') as f:
                        embedding_vector = f[EMBEDDING_KEY][()]
                        
                        if embedding_vector.ndim > 1 and embedding_vector.shape[0] == 1:
                            embedding_vector = embedding_vector.squeeze()
                        
                        all_embeddings.append(embedding_vector)
                        processed_count += 1
                except Exception as e:
                    print(f"File Processing Error (Name: {path}): {e}")

            if all_embeddings:
                feature_matrix = np.vstack(all_embeddings)
                N_sample, D_dimensions = feature_matrix.shape
                
                dataType = os.path.basename(h5_dir)
                filename = f"{dataset}_o{output_len}_{dataType}_consolidated.h5"
                save_path = os.path.join("./data", filename)
                print(f"Processed Matrix: {N_sample} Samples, {D_dimensions} Dimensions")
                
                try:
                    with h5py.File(save_path, 'w') as hf:
                        hf.create_dataset('data', data=feature_matrix, compression='gzip')
                    print(f"Matrix saved to: {save_path}\n")
                except Exception as e:
                    print(f"Error saving file {save_path}: {e}\n")
            else:
                print("Valid embedding vector does not exists.\n")