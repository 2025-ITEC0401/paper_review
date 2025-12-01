# preprocess_high_dim.py

import numpy as np
import pandas as pd
import torch
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

def create_sliding_windows(data_scaled, seq_len, stride=1):
    windows = []
    num_samples = (len(data_scaled) - seq_len) // stride + 1
    for i in tqdm(range(num_samples), desc="Creating sliding windows"):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        windows.append(data_scaled[start_idx:end_idx])
    return np.array(windows)

def main(args):
    print(f"--- Starting PCA preprocessing for {args.dataset_name} ---")

    # 1. 원본 CSV 파일 로드
    file_path = os.path.join(args.root_path, args.data_folder, args.data_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Original data file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 2. 데이터 정규화
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)
    
    # 3. PCA 모델 학습
    print(f"Fitting PCA to reduce {data_scaled.shape[1]} channels to {args.n_components} components...")
    pca = PCA(n_components=args.n_components)
    pca.fit(data_scaled)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA fitting complete. Explained variance ratio: {explained_variance:.4f}")

    # 4. 전체 데이터를 PCA로 변환
    data_pca = pca.transform(data_scaled)

    # 5. 슬라이딩 윈도우 생성
    data_windows = create_sliding_windows(data_pca, args.seq_len) # (N, L, C')
    
    # 6. (N, C', L) 형태로 변환
    final_tensor = torch.tensor(data_windows, dtype=torch.float32).permute(0, 2, 1)
    print(f"Transformed data shape (N, C', L): {final_tensor.shape}\n")

    # 7. 결과 저장
    output_dir = os.path.join(args.root_path, args.data_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    pca_data_path = os.path.join(output_dir, f'{args.dataset_name}_pca{args.n_components}.npy')
    pca_model_path = os.path.join(output_dir, f'{args.dataset_name}_pca{args.n_components}_model.joblib')

    np.save(pca_data_path, final_tensor.numpy())
    joblib.dump(pca, pca_model_path)

    print(f"✅ Preprocessed data saved to: {pca_data_path}")
    print(f"✅ PCA model saved to: {pca_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess High-Dimensional Time Series with PCA")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., traffic)')
    parser.add_argument('--root_path', type=str, default='/hdd/intern/z_timeKD/dataset', help='Root path of datasets')
    parser.add_argument('--data_folder', type=str, required=True, help='Folder containing the data file (e.g., traffic)')
    parser.add_argument('--data_file', type=str, required=True, help='Name of the data file (e.g., traffic.csv)')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length for sliding window')
    parser.add_argument('--n_components', type=int, default=16, help='Number of principal components to keep')
    
    args = parser.parse_args()
    main(args)
