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
import pandas as pd
import warnings

# 모델 파일 로드
from models.PatchTST import Model as PatchTST

# -----------------------------------------------------------------------------
# 0. 시각화 함수
# -----------------------------------------------------------------------------
def visualize_imputation(true, pred, mask, dataset_name, sample_idx=0, channel_idx=0):
    plt.figure(figsize=(12, 6))
    
    # 텐서를 넘파이로 변환
    t_true = true[sample_idx, :, channel_idx].cpu().numpy()
    t_pred = pred[sample_idx, :, channel_idx].cpu().numpy()
    t_mask = mask[sample_idx, :, channel_idx].cpu().numpy()
    
    x_axis = np.arange(len(t_true))
    
    # 1. 원본 그리기
    plt.plot(x_axis, t_true, label='Ground Truth', color='blue', alpha=0.3, linewidth=2)
    
    # 2. 복원된 값 그리기
    masked_indices = np.where(t_mask == 1)[0]
    plt.plot(x_axis, t_pred, label='Imputed Prediction', color='orange', linestyle='--')
    plt.scatter(masked_indices, t_pred[masked_indices], color='red', s=10, label='Masked Points', zorder=5)

    plt.title(f"Imputation Result: {dataset_name} (Sample {sample_idx}, Channel {channel_idx})")
    plt.xlabel("Time Steps")
    plt.ylabel("Value (Standardized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_name = f"Imputation_{dataset_name}.png"
    plt.savefig(save_name)
    plt.close()
    print(f"   -> [시각화 저장 완료]: {save_name}")

# -----------------------------------------------------------------------------
# 1. 지표 계산 함수
# -----------------------------------------------------------------------------
def calc_metrics(pred, true, mask):
    # 마스킹된 부분만 평가
    pred_masked = pred[mask]
    true_masked = true[mask]
    
    if len(true_masked) == 0: return 0.0, 0.0, 0.0

    mse = torch.mean((pred_masked - true_masked) ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred_masked - true_masked)).item()
    
    return mse, rmse, mae

# -----------------------------------------------------------------------------
# 2. 마스킹 함수
# -----------------------------------------------------------------------------
def random_masking(x, mask_ratio, device):
    rand = torch.rand_like(x)
    mask = rand < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked.to(device), mask.to(device)

# -----------------------------------------------------------------------------
# 3. 데이터 로딩
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
    
    scaler = StandardScaler()
    n_train, t_len, n_ch = X_train.shape
    n_test, _, _ = X_test.shape
    
    X_train_flat = X_train.reshape(-1, n_ch)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, t_len, n_ch)
    
    X_test_flat = X_test.reshape(-1, n_ch)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_test, t_len, n_ch)

    return X_train_scaled, X_test_scaled

# -----------------------------------------------------------------------------
# 4. 실행 루프
# -----------------------------------------------------------------------------
def run_experiment(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device: {device} ---")

    X_train, X_test = load_data(args.dataset_name, args.root_path)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float()), batch_size=args.batch_size, shuffle=False)

    # [Task 에러 해결] 모델을 '장기 예측' 모드로 속여서 Imputation 수행
    args.task_name = 'long_term_forecast' 
    args.pred_len = args.seq_len # 입력 길이만큼 전체를 예측(=복원)
    args.label_len = 0 
    args.num_class = 0
    args.c_out = args.enc_in 
    
    model = PatchTST(args).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"\n[Training Start] Mask Ratio: {args.mask_ratio}")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            
            # 랜덤 마스킹
            x_masked, mask = random_masking(batch_x, args.mask_ratio, device)
            
            # None 인자 전달 (Forecasting 모드 호환성)
            outputs = model(x_masked, None, None, None)
            
            # Loss 계산
            loss = criterion(outputs, batch_x) 
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {np.mean(train_losses):.5f}")

    print("\n[Evaluation Start]")
    model.eval()
    
    mse_list, rmse_list, mae_list = [], [], []
    
    with torch.no_grad():
        for i, (batch_x,) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            
            x_masked, mask = random_masking(batch_x, args.mask_ratio, device)
            outputs = model(x_masked, None, None, None)
            
            mse, rmse, mae = calc_metrics(outputs, batch_x, mask)
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)

            if i == 0:
                visualize_imputation(batch_x, outputs, mask, args.dataset_name)

    print("\n" + "="*45)
    print(f" [Final Result] Dataset: {args.dataset_name}")
    print("-" * 45)
    print(f" 1. MSE        : {np.mean(mse_list):.6f}")
    print(f" 2. RMSE       : {np.mean(rmse_list):.6f}")
    print(f" 3. MAE        : {np.mean(mae_list):.6f}")
    print(f" 4. Mask Ratio : {args.mask_ratio:.2f}")
    print("="*45 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--gpu', type=int, default=0)
    
    # 모델 하이퍼파라미터
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    
    # PatchTST 필요 인자들
    parser.add_argument('--pred_len', type=int, default=24)
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
