import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from sktime.datasets._data_io import load_from_tsfile
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering # SpectralClustering 추가
from sklearn.metrics import rand_score, normalized_mutual_info_score

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
# 2. 데이터셋 샘플 시각화
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
# 3. t-SNE 및 Clustering (K-Means + Spectral)
# ---------------------------------------------------------
def plot_tsne(dataset_name, root_path):
    file_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    if not os.path.exists(file_path):
        print(f"[알림] TEST 파일이 없어 TRAIN 파일로 대체합니다.")
        file_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
        if not os.path.exists(file_path): 
            print("[오류] 데이터 파일을 찾을 수 없습니다.")
            return

    print(f"--- {dataset_name} 분석 시작 ---")
    X_df, y = load_from_tsfile(file_path)
    
    n_samples = X_df.shape[0]
    X_flattened = []
    
    # Spectral Clustering은 메모리를 많이 쓰므로 샘플링 필수 (최대 500개)
    limit = 500
    if n_samples > limit:
        indices = np.random.choice(n_samples, limit, replace=False)
        X_df = X_df.iloc[indices]
        y = y[indices]
        n_samples = limit
        
    # 데이터 평탄화 (Samples, Features)
    for i in range(n_samples):
        row_data = np.concatenate([X_df.iloc[i, c].to_numpy() for c in range(X_df.shape[1])])
        X_flattened.append(row_data)
    X_flattened = np.array(X_flattened)

    # -----------------------------------------------------
    # Clustering 1: K-Means
    # -----------------------------------------------------
    n_classes = len(np.unique(y))
    
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    y_pred_kmeans = kmeans.fit_predict(X_flattened)
    
    ri_k = rand_score(y, y_pred_kmeans)
    nmi_k = normalized_mutual_info_score(y, y_pred_kmeans)

    # -----------------------------------------------------
    # Clustering 2: Spectral Clustering (New!)
    # -----------------------------------------------------
    # affinity='nearest_neighbors'가 고차원 데이터에서 보통 더 안정적입니다.
    spectral = SpectralClustering(n_clusters=n_classes, affinity='nearest_neighbors', random_state=42, n_jobs=-1)
    y_pred_spectral = spectral.fit_predict(X_flattened)
    
    ri_s = rand_score(y, y_pred_spectral)
    nmi_s = normalized_mutual_info_score(y, y_pred_spectral)
    
    print(f"Metrics for {dataset_name}:")
    print(f"  [K-Means]  RI: {ri_k:.4f}, NMI: {nmi_k:.4f}")
    print(f"  [Spectral] RI: {ri_s:.4f}, NMI: {nmi_s:.4f}")

    # -----------------------------------------------------
    # t-SNE 시각화
    # -----------------------------------------------------
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    X_embedded = tsne.fit_transform(X_flattened)

    df_plot = pd.DataFrame(X_embedded, columns=['x', 'y'])
    df_plot['class'] = y # 색상은 실제 정답(Ground Truth)으로 표시

    plt.figure(figsize=(10, 8))
    unique_classes = np.unique(y)
    for label in unique_classes:
        subset = df_plot[df_plot['class'] == label]
        plt.scatter(subset['x'], subset['y'], label=label, alpha=0.7, s=40)

    # 제목에 두 모델의 점수를 모두 표시
    title_str = (f"{dataset_name} Visualization\n"
                 f"K-Means -> RI: {ri_k:.4f}, NMI: {nmi_k:.4f}\n"
                 f"Spectral -> RI: {ri_s:.4f}, NMI: {nmi_s:.4f}")
    
    plt.title(title_str, fontsize=13)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_name = f"tsne_{dataset_name}.png"
    plt.savefig(save_name)
    plt.close()
    print(f" -> {save_name} 저장 완료")

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