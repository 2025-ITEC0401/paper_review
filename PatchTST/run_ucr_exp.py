# run_ucr_exp.py (revin=0, individual=1 최종 수정본)

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# sktime에서 로컬 .ts 파일 로더를 가져옵니다.
from sktime.datasets._data_io import load_from_tsfile

from sklearn.model_selection import train_test_split
from models.PatchTST import Model as PatchTST

# 1. 로컬 데이터 로딩 및 전처리 함수
def load_and_prep_data(dataset_name, root_path, batch_size):
    print(f"--- {dataset_name} 로컬 데이터셋 로딩 중... ---")
    
    # 훈련/테스트 데이터 파일 경로
    train_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"오류: {train_file} 또는 {test_file}을 찾을 수 없습니다.")
        return None, None

    # sktime의 로더로 .ts 파일 로드
    X_train_pd, y_train = load_from_tsfile(train_file)
    X_test_pd, y_test = load_from_tsfile(test_file)
    
    # --- sktime(nested pandas)를 (samples, timesteps, channels) 3D Numpy로 직접 변환 ---
    def convert_to_3d_numpy(X_pd):
        n_samples = X_pd.shape[0]
        n_channels = X_pd.shape[1]
        n_timesteps = X_pd.iloc[0, 0].shape[0]
        arr = np.empty((n_samples, n_timesteps, n_channels), dtype=np.float32)
        for i in range(n_samples):
            for j in range(n_channels):
                arr[i, :, j] = X_pd.iloc[i, j].to_numpy()
        return arr

    print("... 3D Numpy 배열로 변환 중 ...")
    X_train_np = convert_to_3d_numpy(X_train_pd)
    X_test_np = convert_to_3d_numpy(X_test_pd)
    print("... 변환 완료 ...")
    # --- [변환 완료] ---

    # PyTorch Tensor로 변환
    X_train_tensor = torch.from_numpy(X_train_np).float()
    X_val_tensor = torch.from_numpy(X_test_np).float()
    
    # DataLoader 생성
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"데이터 준비 완료. Train: {len(X_train_np)} samples, Val: {len(X_test_np)} samples")
    return train_loader, val_loader

# 2. 메인 실행 함수
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PatchTST for UCR/UEA Datasets')
    # 필수 인자
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--enc_in', type=int, required=True)
    
    # 모델 하이퍼파라미터
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    
    # 훈련 관련 인자
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # --- PatchTST가 필요로 하는 기본 설정값들 ---
    parser.add_argument('--task_name', type=str, default='reconstruction')
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--padding_patch', default='end')
    
    # --- [핵심 수정 1] RevIN 비활성화 ---
    parser.add_argument('--revin', type=int, default=0) # 1에서 0으로 변경
    
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    
    # --- [핵심 수정 2] 채널 독립 처리 활성화 ---
    parser.add_argument('--individual', type=int, default=1) # 0에서 1로 변경
    
    parser.add_argument('--embed_type', type=int, default=0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    # --- [기본 설정값 완료] ---

    args = parser.parse_args()
    args.c_out = args.enc_in 
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"--- 사용할 장치: {device} ---")
    
    train_loader, val_loader = load_and_prep_data(args.dataset_name, args.root_path, args.batch_size)
    
    if train_loader is None:
        print(f"--- {args.dataset_name} 데이터 로딩 실패. 훈련을 건너뜁니다. ---")
    else:
        model = PatchTST(args).float().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        print(f"\n--- {args.dataset_name} 훈련 시작 ---")
        
        # 체크포인트 저장 경로 설정
        setting = f'{args.dataset_name}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}'
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        for epoch in range(args.epochs):
            model.train()
            train_loss = []
            for i, (batch_x,) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.to(device)
                
                outputs = model(batch_x) 
                loss = criterion(outputs, batch_x) 
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = []
            with torch.no_grad():
                for i, (batch_x,) in enumerate(val_loader):
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    val_loss.append(loss.item())
            
            print(f"Epoch: {epoch+1}/{args.epochs} | Train Loss: {np.mean(train_loss):.4f} | Val Loss: {np.mean(val_loss):.4f}")

        # 훈련 완료 후 모델 저장
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        print(f"--- {args.dataset_name} 훈련 종료 및 모델 저장 완료 ---")
