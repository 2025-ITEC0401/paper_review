import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sktime.datasets._data_io import load_from_tsfile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import warnings

# 모델 로드
from models.PatchTST import Model as PatchTST

# -----------------------------------------------------------------------------
# 0. 시각화 함수
# -----------------------------------------------------------------------------
def visualize_forecast(true, pred, dataset_name, sample_idx=0, channel_idx=0, mode='Forecast'):
    plt.figure(figsize=(12, 6))
    
    # Numpy 변환 (배치 차원 제거)
    t_true = true[sample_idx, :, channel_idx].cpu().numpy()
    t_pred = pred[sample_idx, :, channel_idx].cpu().numpy()
    
    x_axis = np.arange(len(t_true))
    
    plt.plot(x_axis, t_true, label='Ground Truth', color='blue', marker='.', markersize=4, alpha=0.4)
    plt.plot(x_axis, t_pred, label='Prediction', color='red', linestyle='-', linewidth=1.5, alpha=0.8)

    plt.title(f"[{mode}] {dataset_name} (Sample {sample_idx}, Channel {channel_idx})")
    plt.xlabel("Time Steps")
    plt.ylabel("Value (Standardized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_name = f"Forecast_{dataset_name}.png"
    plt.savefig(save_name)
    plt.close()
    print(f"   -> [Visual saved]: {save_name}")

# -----------------------------------------------------------------------------
# 1. 지표 계산 함수 (MSE, RMSE, MAE, R2)
# -----------------------------------------------------------------------------
def calc_metrics(pred, true):
    # Flatten for sklearn
    pred_np = pred.detach().cpu().numpy().flatten()
    true_np = true.detach().cpu().numpy().flatten()
    
    mse = mean_squared_error(true_np, pred_np)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true_np, pred_np)
    r2 = r2_score(true_np, pred_np)
    
    return mse, rmse, mae, r2

# -----------------------------------------------------------------------------
# 2. 데이터 로딩
# -----------------------------------------------------------------------------
def load_data(dataset_name, root_path):
    train_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    print(f"--- Loading {dataset_name}... ---")
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
    
    # Scaling
    scaler = StandardScaler()
    n_train, t_len, n_ch = X_train.shape
    n_test, _, _ = X_test.shape
    
    X_train_flat = X_train.reshape(-1, n_ch)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, t_len, n_ch)
    
    X_test_flat = X_test.reshape(-1, n_ch)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_test, t_len, n_ch)

    return X_train_scaled, X_test_scaled

# -----------------------------------------------------------------------------
# 3. 실행 루프
# -----------------------------------------------------------------------------
def run_experiment(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device: {device} ---")
    torch.cuda.empty_cache()

    # 1. 데이터 로드
    X_train_raw, X_test_raw = load_data(args.dataset_name, args.root_path)
    
    # 2. [Auto-Config] 데이터 형태 기반 설정
    total_len = X_train_raw.shape[1]
    n_channels = X_train_raw.shape[2]
    
    # 예측 길이는 인자 혹은 기본값(24) 사용. 단, 데이터보다 길면 안됨.
    if args.pred_len >= total_len:
        args.pred_len = int(total_len * 0.2)
        
    seq_len = total_len - args.pred_len
    
    print(f"\n[Auto-Config] Data Length: {total_len}")
    print(f"[Auto-Config] Channels: {n_channels}")

    # Reconstruction 모드로 갈지 Forecast 모드로 갈지 결정하기 위해
    # 일단 모델을 초기화할 때 'reconstruction'이 안전함 (Task Check 후 결정)
    
    # Reconstruction 모드로 설정 시: Input = Total, Output = Total
    # Forecast 모드로 설정 시: Input = seq_len, Output = pred_len
    
    # 여기서는 안전하게 '전체 길이 복원'을 기본 전략으로 가져감 (Task 미지원 대비)
    args.seq_len = total_len
    args.pred_len = total_len
    args.enc_in = n_channels
    args.c_out = n_channels 
    args.context_window = total_len
    args.task_name = 'reconstruction' # Default fallback
    args.label_len = 0
    args.num_class = 0

    # 데이터셋 구성 (Reconstruction용: Input=Total, Label=Total)
    train_x = torch.from_numpy(X_train_raw).float()
    test_x = torch.from_numpy(X_test_raw).float()
    
    train_loader = DataLoader(TensorDataset(train_x, train_x), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_x), batch_size=args.batch_size, shuffle=False)

    print(f"--- Model Init (seq_len={args.seq_len}) ---")
    model = PatchTST(args).float().to(device)
    
    # [수정됨] W_pos 강제 수정 코드 삭제됨. 
    # 모델이 알아서 초기화한 값을 신뢰합니다.

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"\n[Training Start] Mode: Reconstruction (Forecasting Proxy)")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Reconstruction Input
            outputs = model(batch_x, None, None, None)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {np.mean(train_losses):.5f}")

    print("\n[Evaluation Start]")
    model.eval()
    
    total_mse, total_rmse, total_mae, total_r2 = [], [], [], []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x, None, None, None)
            
            # 지표 계산 및 시각화
            mse, rmse, mae, r2 = calc_metrics(outputs, batch_y)
            total_mse.append(mse)
            total_rmse.append(rmse)
            total_mae.append(mae)
            total_r2.append(r2)

            if i == 0:
                visualize_forecast(batch_y, outputs, args.dataset_name, mode='Reconstruction')
            
            torch.cuda.empty_cache()

    print("\n" + "="*45)
    print(f" [Final Result] Dataset: {args.dataset_name}")
    print("-" * 45)
    print(f" 1. MSE  : {np.mean(total_mse):.6f}")
    print(f" 2. RMSE : {np.mean(total_rmse):.6f}")
    print(f" 3. MAE  : {np.mean(total_mae):.6f}")
    print(f" 4. R2   : {np.mean(total_r2):.6f}")
    print("="*45 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pred_len', type=int, default=24) 
    
    # PatchTST Defaults
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
    run_experiment(args)
