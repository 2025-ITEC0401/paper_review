import h5py
import numpy as np
import os
from glob import glob

EMBEDDING_KEY = 'embeddings'

for output_len in [24, 36, 48, 96, 192]:
    h5_dir_list = [f'./data/{output_len}/train', f'./data/{output_len}/val']
    for h5_dir in h5_dir_list:
        h5_files = glob(os.path.join(h5_dir, '*.h5'))
        if not h5_files:
            print(f"h5 file cannot be found. Check the path. {h5_dir}")
            continue
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
            print(f"Complete! (Output_len: {output_len}, dir: {h5_dir})")
            print(f"Processed Matrix: {N_sample} Samples, {D_dimensions} Dimensions.")
        else:
            print("Valid embedding vector does not exists.")