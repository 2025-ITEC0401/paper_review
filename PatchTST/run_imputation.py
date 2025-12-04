import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import math
from sktime.datasets._data_io import load_from_tsfile
from sklearn.preprocessing import StandardScaler
from models.PatchTST import Model as PatchTST

# -----------------------------------------------------------------------------
# 1. 지표 계산 함수
# -----------------------------------------------------------------------------
def calc_metrics(pred, true, mask):
    """
    Mask 처리된 부분(결측치였던 부분)에 대해서만 오차를 계산합니다.
    """
    # 마스킹된 위치의 데이터만 추출
    pred_masked = pred[mask]
    true_masked = true[mask]
    
    if len(true_masked) == 0:
        return 0.0, 0.0, 0.0

    mse = torch.mean((pred_masked - true_masked) ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred_masked - true_masked)).item()
    
    return mse, rmse, mae

# -----------------------------------------------------------------------------
# 2. 마스킹 함수 (데이터에 구멍 뚫기)
# -----------------------------------------------------------------------------
def random_masking(x, mask_ratio, device):
    """
    Input: (Batch, Time, Channel)
    Operation: mask_ratio 확률로 0으로 만듦
    Output: Masked Input, Mask Matrix (1=masked, 0=kept)
    """
    # 0~1 사이 랜덤 값 생성
    rand = torch.rand_like(x)
    # mask_ratio보다 작은 부분을 True(Masked)로 설정
    mask = rand < mask_ratio
    
    # 마스킹된 입력 데이터 생성 (가려진 부분은 0으로 채움)
    x_masked = x.clone()
    x_masked[mask] = 0
    
    return x_masked.to(device), mask.to(device)

# -----------------------------------------------------------------------------
# 3. 데이터 로딩 및 전처리
# -----------------------------------------------------------------------------
def load_data(dataset_name, root_path):
    train_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    
    # 데이터 로드 (sktime)
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    import pandas as pd # pandas import 추가
    
    print(f"--- Loading {dataset_name} ... ---")
    X_train_df, _ = load_from_tsfile(train_file)
    X_test_df, _ = load_from_tsfile(test_file)

    def to_numpy_3d(df):
        n_samples = df.shape[0]
        n_channels = df.shape[1]
        n_timesteps = df.iloc[0, 0].shape[0]
        arr = np.empty((n_samples, n_timesteps, n_channels), dtype=np.float32)
        for i in range(n_samples):
            for j in range(n_channels):
                arr[i, :, j] = df.iloc[i, j].to_numpy()
        return arr

    X_train = to_numpy_3d(X_train_df)
    X_test = to_numpy_3d(X_test_df)
    
    # 데이터 정규화 (StandardScaler) - Imputation 성능에 중요
    # (Samples, Time, Channel) -> (Samples * Time, Channel) 
    n_train, t_len, n_ch = X_train.shape
    n_test, _, _ = X_test.shape
    
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_ch)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, t_len, n_ch)
    
    X_test_flat = X_test.reshape(-1, n_ch)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_test, t_len, n_ch)

    return X_train_scaled, X_test_scaled

# -----------------------------------------------------------------------------
# 4. 학습 및 평가 루프
# -----------------------------------------------------------------------------
def run_imputation_experiment(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 1. 데이터 로드
    X_train, X_test = load_data(args.dataset_name, args.root_path)
    
    # DataLoader 생성
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float()), 
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float()), 
                             batch_size=args.batch_size, shuffle=False)

    # 2. 모델 초기화
    # PatchTST Imputation 모드 설정
    args.task_name = 'imputation'
    args.num_class = 0
    args.c_out = args.enc_in # 출력 채널 = 입력 채널
    
    model = PatchTST(args).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"\n[Start Training] Dataset: {args.dataset_name}, Mask Ratio: {args.mask_ratio}")

    # -----------------------------------------------------
    # 학습 (Training) - 마스킹된 데이터를 복원하도록 학습
    # -----------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            
            # 랜덤 마스킹 적용
            x_masked, mask = random_masking(batch_x, args.mask_ratio, device)
            
            # 모델 예측 (입력: 마스킹된 데이터)
            outputs = model(x_masked)
            
            # Loss 계산: 마스킹된 부분과 원본의 차이만 계산 (혹은 전체 재구성 오차)
            # 일반적으로 Imputation 학습 시에는 전체 혹은 마스크 부분 Loss를 씀
            loss = criterion(outputs, batch_x) 
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        print(f"Epoch: {epoch+1}/{args.epochs} | Train Loss: {np.mean(train_loss):.4f}")

    # -----------------------------------------------------
    # 평가 (Evaluation) - Test 셋으로 지표 계산
    # -----------------------------------------------------
    print("\n[Start Evaluation]")
    model.eval()
    
    total_mse = []
    total_rmse = []
    total_mae = []
    
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            
            # 테스트 데이터에 마스킹 적용 (이 부분을 복원해야 함)
            x_masked, mask = random_masking(batch_x, args.mask_ratio, device)
            
            # 예측
            outputs = model(x_masked)
            
            # 지표 계산 (마스킹된 부분만 비교)
            mse, rmse, mae = calc_metrics(outputs, batch_x, mask)
            
            total_mse.append(mse)
            total_rmse.append(rmse)
            total_mae.append(mae)

    # 최종 결과 출력
    avg_mse = np.mean(total_mse)
    avg_rmse = np.mean(total_rmse)
    avg_mae = np.mean(total_mae)
    
    print("\n" + "="*40)
    print(f" Dataset: {args.dataset_name}")
    print(f" Mask Ratio used: {args.mask_ratio}")
    print("-" * 40)
    print(f" 1. MSE        : {avg_mse:.6f}")
    print(f" 2. RMSE       : {avg_rmse:.6f}")
    print(f" 3. MAE        : {avg_mae:.6f}")
    print(f" 4. Mask Ratio : {args.mask_ratio:.2f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='결측치 비율 (0.0 ~ 1.0)')
    parser.add_argument('--gpu', type=int, default=0)
    
    # 모델 파라미터
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    
    # PatchTST 기본 인자 (Dummy)
    parser.add_argument('--pred_len', type=int, default=24) # dummy
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--fc_dropout', type=float, default=0.0)
    parser.add_argument('--padding_patch', default='end')
    parser.add_argument('--revin', type=int, default=0)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--individual', type=int, default=1)
    parser.add_argument('--embed_type', type=int, default=0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    
    args = parser.parse_args()
    
    run_imputation_experiment(args)