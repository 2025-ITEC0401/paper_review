# ./src/data_loader.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
from sktime.datasets import load_from_tsfile # sktime 라이브러리 임포트

def load_uea_dataset(dataset_name, root_path):
    """
    UEA/UCR 데이터셋(.ts 파일)을 로드하고 (N, C, L) 형태의 텐서와 레이블을 반환합니다.
    """
    train_path = os.path.join(root_path, dataset_name, f'{dataset_name}_TRAIN.ts')
    test_path = os.path.join(root_path, dataset_name, f'{dataset_name}_TEST.ts')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train or test file not found for dataset {dataset_name} in {os.path.join(root_path, dataset_name)}")

    X_train, y_train = load_from_tsfile(train_path, return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(test_path, return_data_type="numpy3d")

    # 비지도 학습이므로 train/test 데이터를 합칩니다.
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    
    # (N, C, L) 형태로 이미 로드되므로 permute 필요 없음
    print(f"Loaded {dataset_name}. Shape: {X_all.shape}, Labels shape: {y_all.shape}")
    
    # 데이터 스케일링: 각 시계열 샘플별로 독립적으로 수행
    # (N, C, L) -> (N*C, L) 형태로 잠시 변경 후 스케일링
    N, C, L = X_all.shape
    X_all_reshaped = X_all.reshape(N * C, L)
    scaler = StandardScaler()
    X_all_scaled_reshaped = scaler.fit_transform(X_all_reshaped.T).T
    X_all_scaled = X_all_scaled_reshaped.reshape(N, C, L)
        
    return torch.from_numpy(X_all_scaled).float(), y_all

def load_and_preprocess_data(dataset_name, root_path, seq_len=96, stride=1):
    """
    데이터셋을 로드하고 전처리합니다.
    - 'BasicMotions', 'DuckDuckGeese', 'Epilepsy' 등: UEA 데이터셋 로더 호출
    - 'ETTh1', 'traffic' 등: 기존 CSV/PCA 로더 호출
    """
    # 새로운 UEA 데이터셋 리스트
    uea_datasets = ['BasicMotions', 'DuckDuckGeese', 'Epilepsy', 'Libras', 'HandMovementDirection',
                    'AtrialFibrillation', 'StandWalkJump', 'ArticularyWordRecognition',
                    'NATOPS', 'PenDigits', 'UWaveGestureLibrary', 'PEMS-SF'] 

    if dataset_name in uea_datasets:
        return load_uea_dataset(dataset_name, root_path)

    # --- 기존 코드 (CSV/PCA 데이터 처리) ---
    data_map = {
        'ETTh1': {'dir': 'ETT-small', 'file': 'ETTh1.csv'},
        'traffic': {'dir': 'traffic', 'file': 'traffic.csv'},
        'exchange': {'dir': 'exchange_rate', 'file': 'exchange_rate.csv'},
        'electricity': {'dir': 'electricity', 'file': 'electricity.csv'},
        'HVAC': {'dir': 'HVAC', 'file': 'HVAC.csv'}
    }

    if dataset_name not in data_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please add it to data_map or the uea_datasets list in data_loader.py")

    data_folder = data_map[dataset_name]['dir']
    
    preprocessed_file_path = os.path.join(root_path, data_folder, f'{dataset_name}_pca16.npy')

    if os.path.exists(preprocessed_file_path):
        print(f"✅ Found preprocessed PCA file. Loading from: {preprocessed_file_path}")
        data = np.load(preprocessed_file_path)
        print(f"Finished. Shape: {data.shape}")
        return torch.from_numpy(data).float(), None # 레이블 없음

    print(f"⚠️ Preprocessed file not found. Loading and processing from original CSV...")
    
    file_path = os.path.join(root_path, data_folder, data_map[dataset_name]['file'])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)

    windows = []
    num_samples = (len(data_scaled) - seq_len) // stride + 1
    
    for i in tqdm(range(num_samples), desc=f"Creating sliding windows for {dataset_name}"):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        windows.append(data_scaled[start_idx:end_idx])

    data_windows = np.array(windows)
    data_tensor = torch.tensor(data_windows, dtype=torch.float32).permute(0, 2, 1)

    print(f"Finished. Shape: {data_tensor.shape}")
    return data_tensor, None # 레이블 없음