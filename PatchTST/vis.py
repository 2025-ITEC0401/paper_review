import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from sktime.datasets._data_io import load_from_tsfile
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# 1. 학습 로그(Loss) 시각화
# ---------------------------------------------------------
def plot_loss_from_log(log_file):
    if not os.path.exists(log_file):
        print(f"[오류] {log_file} 파일을 찾을 수 없습니다.")
        return

    train_losses, val_losses, epochs = [], [], []
    current_dataset = "Unknown"
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if "훈련 시작" in line:
            current_dataset = line.split(":")[1].strip().split("(")[0].strip()
            train_losses, val_losses, epochs = [], [], []
            print(f"\n[{current_dataset}] Loss 그래프 생성 준비...")

        if "Train Loss:" in line and "Val Loss:" in line:
            try:
                parts = line.split("|")
                epochs.append(int(parts[0].split(":")[1].split("/")[0]))
                train_losses.append(float(parts[1].split(":")[1]))
                val_losses.append(float(parts[2].split(":")[1]))
            except: continue

        if "훈련 종료" in line or "모든 훈련 완료" in line:
            if len(epochs) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(epochs, train_losses, label='Train', marker='.')
                plt.plot(epochs, val_losses, label='Val', marker='.')
                plt.title(f'{current_dataset} Training Progress')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"loss_{current_dataset}.png")
                plt.close()

# ---------------------------------------------------------
# 2. 데이터셋 샘플 시각화 (Raw Data)
# ---------------------------------------------------------
def plot_dataset_sample(dataset_name, root_path):
    file_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    if not os.path.exists(file_path):
        print(f"[오류] {file_path} 없음")
        return

    print(f"--- {dataset_name} 데이터 로딩 중... ---")
    X, y = load_from_tsfile(file_path)
    sample = X.iloc[0, :]
    
    plt.figure(figsize=(12, 4))
    for i in range(min(sample.shape[0], 3)):
        plt.plot(sample.iloc[i].to_numpy(), label=f'Channel {i}')
    plt.title(f'{dataset_name} - Sample 0')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"sample_{dataset_name}.png")
    plt.close()
    print(f" -> sample_{dataset_name}.png 저장 완료")

# ---------------------------------------------------------
# 3. t-SNE 시각화 (요청하신 선 연결 스타일 적용)
# ---------------------------------------------------------
def plot_tsne(dataset_name, root_path):
    file_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    if not os.path.exists(file_path):
        print(f"[오류] {file_path} 없음. TRAIN 파일로 대체 시도...")
        file_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
        if not os.path.exists(file_path): return

    print(f"--- {dataset_name} t-SNE 계산 중 (시간이 걸릴 수 있음)... ---")
    X_df, y = load_from_tsfile(file_path)
    
    # 데이터 전처리: (Samples, Channels, Time) -> (Samples, Flattened Features)
    # t-SNE는 2차원 입력을 받으므로 시계열을 1줄로 폅니다.
    n_samples = X_df.shape[0]
    X_flattened = []
    
    # 너무 많으면 샘플링 (속도 위해 최대 500개만)
    limit = 500
    if n_samples > limit:
        indices = np.random.choice(n_samples, limit, replace=False)
        X_df = X_df.iloc[indices]
        y = y[indices]
        n_samples = limit
        
    for i in range(n_samples):
        # 각 채널의 데이터를 이어붙임
        row_data = np.concatenate([X_df.iloc[i, c].to_numpy() for c in range(X_df.shape[1])])
        X_flattened.append(row_data)
    X_flattened = np.array(X_flattened)

    # t-SNE 실행
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    X_embedded = tsne.fit_transform(X_flattened)

    # --- [요청하신 시각화 스타일 적용] ---
    df_plot = pd.DataFrame(X_embedded, columns=['x', 'y'])
    df_plot['class'] = y

    plt.figure(figsize=(12, 10))
    classes = sorted(df_plot['class'].unique())
    
    # 색상맵 자동 생성
    cmap = plt.get_cmap('tab10')
    
    for idx, cls in enumerate(classes):
        subset = df_plot[df_plot['class'] == cls]
        # X축 좌표 기준으로 정렬 (선이 꼬이지 않게)
        subset = subset.sort_values(by='x')
        
        color = cmap(idx % 10)
        plt.plot(subset['x'], subset['y'], marker='o', linestyle='-', 
                 linewidth=1.5, markersize=6, alpha=0.8, label=cls, color=color)

    plt.title(f'{dataset_name} - t-SNE (Connected by Class)', fontsize=15)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_name = f"tsne_{dataset_name}.png"
    plt.savefig(save_name)
    plt.close()
    print(f" -> {save_name} 저장 완료!")

# ---------------------------------------------------------
# 메인 실행부
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['loss', 'data', 'tsne'], help='loss / data / tsne')
    parser.add_argument('--dataset', type=str, default='AtrialFibrillation')
    parser.add_argument('--root_path', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--log', type=str, default='experiment.log')
    
    args = parser.parse_args()

    if args.mode == 'loss':
        plot_loss_from_log(args.log)
    elif args.mode == 'data':
        plot_dataset_sample(args.dataset, args.root_path)
    elif args.mode == 'tsne':
        plot_tsne(args.dataset, args.root_path)