import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from sktime.datasets._data_io import load_from_tsfile
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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
# 3. t-SNE 및 RI, NMI 계산 (수정됨)
# ---------------------------------------------------------
def plot_tsne(dataset_name, root_path):
    file_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    if not os.path.exists(file_path):
        print(f"[알림] TEST 파일이 없어 TRAIN 파일로 대체합니다.")
        file_path = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
        if not os.path.exists(file_path): 
            print("[오류] 데이터 파일을 찾을 수 없습니다.")
            return

    print(f"--- {dataset_name} 데이터 로딩 및 분석 중... ---")
    X_df, y = load_from_tsfile(file_path)
    
    n_samples = X_df.shape[0]
    X_flattened = []
    
    # 데이터가 너무 많으면 500개만 샘플링 (속도 최적화)
    limit = 500
    if n_samples > limit:
        indices = np.random.choice(n_samples, limit, replace=False)
        X_df = X_df.iloc[indices]
        y = y[indices]
        n_samples = limit
        
    # (Samples, Time, Channel) -> (Samples, Features) 평탄화
    for i in range(n_samples):
        row_data = np.concatenate([X_df.iloc[i, c].to_numpy() for c in range(X_df.shape[1])])
        X_flattened.append(row_data)
    X_flattened = np.array(X_flattened)

    # -----------------------------------------------------
    # [추가됨] RI, NMI 계산을 위한 K-Means 클러스터링
    # -----------------------------------------------------
    n_classes = len(np.unique(y)) # 실제 클래스 개수 파악
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_flattened) # 원본 특징으로 클러스터링 수행

    ri_score = rand_score(y, y_pred)
    nmi_score = normalized_mutual_info_score(y, y_pred)
    
    print(f"Metrics for {dataset_name}:")
    print(f"  > RI  (Rand Index) : {ri_score:.4f}")
    print(f"  > NMI (Norm MI)    : {nmi_score:.4f}")
    # -----------------------------------------------------

    # t-SNE 실행 (시각화용)
    print("... t-SNE 변환 중 ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    X_embedded = tsne.fit_transform(X_flattened)

    # 시각화 (Scatter Plot)
    df_plot = pd.DataFrame(X_embedded, columns=['x', 'y'])
    df_plot['class'] = y

    plt.figure(figsize=(10, 8))
    unique_classes = np.unique(y)
    for label in unique_classes:
        subset = df_plot[df_plot['class'] == label]
        plt.scatter(subset['x'], subset['y'], label=label, alpha=0.7, s=40)

    # 제목에 점수 표시
    plt.title(f'{dataset_name}\nRI: {ri_score:.4f}, NMI: {nmi_score:.4f}', fontsize=14)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_name = f"tsne_{dataset_name}.png"
    plt.savefig(save_name)
    plt.close()
    print(f" -> {save_name} 저장 완료 (이미지 제목에 점수 포함됨)")

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