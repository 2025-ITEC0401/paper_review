import argparse
import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, normalized_mutual_info_score

# 시스템 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader import Dataset_Custom
from layers.PatchTST_backbone import PatchTST_backbone

def main(args):
    # --- 2x2 격자 형태의 서브플롯 그림을 생성 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Intra-Dataset T-SNE Clustering with RI and NMI Scores', fontsize=20)
    
    # 그래프를 그릴 위치를 쉽게 다루기 위해 1차원 배열로 만듭니다.
    ax_flat = axes.flatten()

    for i, name in enumerate(args.dataset_names):
        print(f"Processing dataset: {name}...")
        
        # 1. 데이터셋 길이를 정확하게 설정합니다.
        if name == 'BasicMotions':
            seq_len = 100
        elif name == 'Epilepsy':
            seq_len = 206
        elif name == 'Libras':
            seq_len = 45
        elif name == 'HandMovementDirection':
            seq_len = 400
        else:
            raise ValueError(f"Unknown dataset or seq_len not defined for: {name}")

        # 2. 데이터 로더 설정
        dataset = Dataset_Custom(root_path=args.root_path, flag='test', data_path=name, size=[0,0,0])
        true_labels = dataset.data_y
        
        # 3. 모델 초기화
        class Configs:
            def __init__(self, seq_len, c_in):
                self.patch_len = 16
                self.stride = 8
                self.revin = False
                self.d_model = 128
                self.n_heads = 8
                self.c_in = c_in
                self.context_window = seq_len
                self.target_window = 0
                self.n_layers = 3
                self.dropout = 0.1
                self.act = "gelu"

        configs = Configs(seq_len, dataset.data_x.shape[-1])
        model = PatchTST_backbone(c_in=configs.c_in, context_window=configs.context_window, 
                                  target_window=configs.target_window, patch_len=configs.patch_len, 
                                  stride=configs.stride, d_model=configs.d_model, n_heads=configs.n_heads,
                                  n_layers=configs.n_layers, dropout=configs.dropout, act=configs.act)
        model.eval()

        # 4. 특징 추출
        with torch.no_grad():
            data_x = torch.FloatTensor(dataset.data_x).permute(0, 2, 1)
            embeddings = model(data_x)
            embeddings = embeddings.mean(dim=(1, 3))
            embeddings_np = embeddings.cpu().numpy()

        # 5. K-Means 클러스터링
        n_clusters = len(np.unique(true_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(embeddings_np)
        
        # 6. RI, NMI 점수 계산
        ri_score = rand_score(true_labels, predicted_labels)
        nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
        print(f"  Scores for {name} -> RI: {ri_score:.4f}, NMI: {nmi_score:.4f}")

        # 7. T-SNE 차원 축소
        print(f"  Running T-SNE for {name}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1), max_iter=1000)
        tsne_results = tsne.fit_transform(embeddings_np)
        
        # 8. 시각화
        ax = ax_flat[i]
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
        
        ax.set_title(f'Clustering within {name}\nRI: {ri_score:.4f} | NMI: {nmi_score:.4f}', fontsize=14)
        ax.set_xlabel('T-SNE Dimension 1')
        ax.set_ylabel('T-SNE Dimension 2')
        
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)

    # 남는 서브플롯이 있다면 보이지 않게 처리합니다.
    for j in range(len(args.dataset_names), len(ax_flat)):
        fig.delaxes(ax_flat[j])

    # --- 결과 이미지 저장 ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_clustering_result.png')
    plt.savefig(save_path)
    print(f"\nClustering visualization saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Clustering and Visualization')
    parser.add_argument('--root_path', type=str, required=True, help='root path of the dataset files')
    parser.add_argument('--dataset_names', nargs='+', required=True, help='list of dataset names')
    args = parser.parse_args()
    
    main(args)
