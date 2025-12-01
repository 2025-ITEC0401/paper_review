import h5py
import numpy as np
import os
from glob import glob

EMBEDDING_KEY = 'embeddings'
# DATASET_LIST = ['ArticularyWordRecognition', 'AtrialFibrillation', 'NATOPS', 'PenDigits', 'PEMS-SF', 'StandWalkJump', 'UWaveGestureLibrary']
# OUTPUT_LIST = [24, 36, 48, 96, 192]
DATASET_LIST = ['ArticularyWordRecognition', 'AtrialFibrillation', 'NATOPS', 'PenDigits', 'StandWalkJump', 'UWaveGestureLibrary']
OUTPUT_LIST = [24]
TYPE = ['train', 'val']
idx = 1

print("\n\n============= Merge .h5 files =============")

for dataset in DATASET_LIST:
    for output_len in OUTPUT_LIST:
        for tp in TYPE:
            h5_dir = f'./data/{dataset}/{output_len}/{tp}'
            h5_files = glob(os.path.join(h5_dir, '*.h5'))
            if not h5_files:
                print(f"({idx}/{len(DATASET_LIST) * len(OUTPUT_LIST) * len(TYPE)}) h5 file cannot be found. Check the path. {h5_dir}")
                idx += 1
                continue
            print(f"\n({idx}/{len(DATASET_LIST) * len(OUTPUT_LIST) * len(TYPE)}) Dataset Name: {dataset}  |  Output Length: {output_len}  |  Path: {h5_dir}")
            print(f"    The number of h5 Files: {len(h5_files)}")

            all_embeddings = []
            processed_count = 0
            for path in h5_files:
                try:
                    if not os.path.exists(path):
                        print(f"File not found: {path}")
                        continue

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
                print(f"({idx}/{len(DATASET_LIST) * len(OUTPUT_LIST) * len(TYPE)}) Processed Matrix: {N_sample} Samples, {D_dimensions} Dimensions")
                
                try:
                    with h5py.File(save_path, 'w') as hf:
                        hf.create_dataset('embeddings', data=feature_matrix, compression='gzip')
                    print(f"({idx}/{len(DATASET_LIST) * len(OUTPUT_LIST) * len(TYPE)}) Matrix saved to: {save_path}\n")
                    idx += 1
                except Exception as e:
                    print(f"({idx}/{len(DATASET_LIST) * len(OUTPUT_LIST) * len(TYPE)}) Error saving file {save_path}: {e}\n")
                    idx += 1
            else:
                print(f"({idx}/{len(DATASET_LIST) * len(OUTPUT_LIST) * len(TYPE)}) Valid embedding vector does not exists.\n")
                idx += 1