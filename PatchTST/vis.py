import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse
from sktime.datasets._data_io import load_from_tsfile

# ---------------------------------------------------------
# 1. 학습 로그(Loss) 시각화 함수
# ---------------------------------------------------------
def plot_loss_from_log(log_file):
    if not os.path.exists(log_file):
        print(f"[오류] {log_file} 파일을 찾을 수 없습니다.")
        return

    train_losses = []
    val_losses = []
    epochs = []

    print(f"--- {log_file} 분석 중... ---")
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    current_dataset = "Unknown"
    
    for line in lines:
        # 데이터셋 이름 찾기 (구분용)
        if "훈련 시작" in line:
            current_dataset = line.split(":")[1].strip().split("(")[0].strip()
            # 새 데이터셋 시작하면 이전 기록 초기화 (원하면 주석 처리 가능)
            train_losses = []
            val_losses = []
            epochs = []
            print(f"\n[{current_dataset}] 그래프 생성 준비...")

        # Loss 패턴 찾기
        # 예: Epoch: 1/20 | Train Loss: 0.1234 | Val Loss: 0.2345
        if "Train Loss:" in line and "Val Loss:" in line:
            try:
                parts = line.split("|")
                epoch_part = parts[0].split(":")[1].strip() # 1/20
                train_part = parts[1].split(":")[1].strip() # 0.1234
                val_part = parts[2].split(":")[1].strip()   # 0.2345
                
                cur_epoch = int(epoch_part.split("/")[0])
                train_loss = float(train_part)
                val_loss = float(val_part)

                epochs.append(cur_epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            except:
                continue

        # 훈련 종료 시 그래프 저장
        if "훈련 종료" in line or "모든 훈련 완료" in line:
            if len(epochs) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(epochs, train_losses, label='Train Loss', marker='.')
                plt.plot(epochs, val_losses, label='Validation Loss', marker='.')
                plt.title(f'{current_dataset} Training Progress')
                plt.xlabel('Epochs')
                plt.ylabel('MSE Loss')
                plt.legend()
                plt.grid(True)
                
                save_name = f"result_{current_dataset}.png"
                plt.savefig(save_name)
                print(f" -> 저장 완료: {save_name}")
                plt.close()

# ---------------------------------------------------------
# 2. 데이터셋 샘플 시각화 함수
# ---------------------------------------------------------
def plot_dataset_sample(dataset_name, root_path):
    file_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    if not os.path.exists(file_path):
        print(f"[오류] {file_path} 없음")
        return

    print(f"--- {dataset_name} 데이터 로딩 및 시각화 중... ---")
    try:
        X, y = load_from_tsfile(file_path)
        
        # 첫 번째 샘플 가져오기
        sample = X.iloc[0, :] # 모든 채널
        n_channels = sample.shape[0]
        
        plt.figure(figsize=(12, 4))
        for i in range(min(n_channels, 3)): # 최대 3개 채널만 그림
            channel_data = sample.iloc[i].to_numpy()
            plt.plot(channel_data, label=f'Channel {i}')
            
        plt.title(f'{dataset_name} - Sample 0 (First 3 Channels)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"sample_{dataset_name}.png")
        print(f" -> 저장 완료: sample_{dataset_name}.png")
        plt.close()
    except Exception as e:
        print(f"Error reading {dataset_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='loss', help='loss: 로그기반 그래프, data: 데이터 샘플 확인')
    parser.add_argument('--log', type=str, default='experiment.log', help='로그 파일 경로')
    parser.add_argument('--data_root', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--dataset', type=str, default='AtrialFibrillation')
    args = parser.parse_args()

    if args.mode == 'loss':
        plot_loss_from_log(args.log)
    elif args.mode == 'data':
        plot_dataset_sample(args.dataset, args.data_root)