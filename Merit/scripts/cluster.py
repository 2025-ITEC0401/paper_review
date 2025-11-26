# ./cluster.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
import umap
import argparse
import os
import time

def main(args):
    print(f"--- Starting Clustering and Evaluation for {args.dataset_name} ---")
    
    # --- 결과 저장 경로 설정 ---
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    scores_file_path = os.path.join(results_dir, f'{args.dataset_name}_clustering_scores.txt')
    # 이전 결과 파일이 있으면 삭제 (새 실행 시 덮어쓰기 위함)
    if os.path.exists(scores_file_path):
        os.remove(scores_file_path)
    # -------------------------

    # --- 1. 데이터 로드 ---
    reps_path = f'./saved_reps/{args.dataset_name}_representations.npy'
    labels_path = f'./saved_reps/{args.dataset_name}_labels.npy'

    if not os.path.exists(reps_path):
        print(f"❌ Representation file not found: {reps_path}")
        return
        
    if not os.path.exists(labels_path):
        print(f"❌ Label file not found: {labels_path}")
        return
        
    representations = np.load(reps_path)
    true_labels = np.load(labels_path)
    unique_labels, true_labels_int = np.unique(true_labels, return_inverse=True)
    n_clusters = len(unique_labels)
    
    print(f"Found {n_clusters} unique classes in the dataset.")
    print(f"✅ Loaded representations with shape: {representations.shape}")
    print(f"✅ Loaded true labels with shape: {true_labels.shape}")

    # --- 2. 클러스터링 수행 ---
    clustering_results = {}

    # K-Means
    print(f"\nPerforming K-Means clustering (k={n_clusters})...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(representations)
    clustering_results['K-Means'] = kmeans_labels
    print(f"K-Means finished in {time.time() - start_time:.2f} seconds.")

    # Spectral Clustering
    print(f"\nPerforming Spectral Clustering (k={n_clusters})...")
    start_time = time.time()
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                  affinity='nearest_neighbors', n_neighbors=10, n_init=10) 
    try:
        spectral_labels = spectral.fit_predict(representations)
        clustering_results['Spectral'] = spectral_labels
        print(f"Spectral Clustering finished in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"⚠️ Spectral Clustering failed: {e}. Skipping.")
        clustering_results['Spectral'] = None # 실패 시 None 저장


    # --- 3. 성능 평가 및 파일 저장 ---
    print("\n--- Clustering Performance ---")
    # 파일 열기 (append 모드)
    with open(scores_file_path, 'a') as f:
        f.write(f"Clustering Performance for Dataset: {args.dataset_name}\n")
        f.write("="*40 + "\n")
        
        for method, pred_labels in clustering_results.items():
            if pred_labels is None: # Spectral Clustering 실패 시 건너뛰기
                 print(f"[{method}] - Skipped due to error during clustering.")
                 f.write(f"[{method}]\n  - Skipped due to error during clustering.\n\n")
                 continue

            ri_score_val = rand_score(true_labels_int, pred_labels)
            ari_score_val = adjusted_rand_score(true_labels_int, pred_labels)
            nmi_score_val = normalized_mutual_info_score(true_labels_int, pred_labels)
            
            # 터미널 출력
            print(f"[{method}]")
            print(f"  Rand Index (RI)           : {ri_score_val:.4f}")
            print(f"  Adjusted Rand Index (ARI) : {ari_score_val:.4f}")
            print(f"  Normalized Mutual Info (NMI): {nmi_score_val:.4f}")

            # 파일 저장
            f.write(f"[{method}]\n")
            f.write(f"  Rand Index (RI)           : {ri_score_val:.4f}\n")
            f.write(f"  Adjusted Rand Index (ARI) : {ari_score_val:.4f}\n")
            f.write(f"  Normalized Mutual Info (NMI): {nmi_score_val:.4f}\n\n")
            
    print("----------------------------")
    print(f"✅ Performance scores saved to {scores_file_path}\n")
    # ---------------------------------

    # --- 4. 차원 축소 ---
    dimensionality_reduction_results = {}

    # t-SNE
    print("Performing t-SNE for visualization (might take a while)...")
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(representations)-1))
    tsne_results = tsne.fit_transform(representations)
    dimensionality_reduction_results['t-SNE'] = tsne_results
    print(f"t-SNE finished in {time.time() - start_time:.2f} seconds.")

    # UMAP
    print("Performing UMAP for visualization...")
    start_time = time.time()
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_results = reducer.fit_transform(representations)
    dimensionality_reduction_results['UMAP'] = umap_results
    print(f"UMAP finished in {time.time() - start_time:.2f} seconds.")


    # --- 5. 결과 시각화 ---
    print("\nPlotting clustering results...")
    num_methods = len(clustering_results) 
    num_dr = len(dimensionality_reduction_results)
    
    fig, axes = plt.subplots(num_dr, num_methods + 1, figsize=(8 * (num_methods + 1), 7 * num_dr), squeeze=False) # squeeze=False 추가
    fig.suptitle(f'Clustering Visualization for {args.dataset_name}', fontsize=20, fontweight='bold')

    cmap = plt.cm.viridis 

    for i, (dr_name, dr_data) in enumerate(dimensionality_reduction_results.items()):
        
        col_idx = 0 # 열 인덱스 초기화
        for method_name, pred_labels in clustering_results.items():
            ax = axes[i, col_idx] 
            
            if pred_labels is None: # 클러스터링 실패 시
                 ax.text(0.5, 0.5, 'Clustering Failed', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                 ax.set_title(f'{method_name} ({dr_name})\nSkipped', fontsize=14)
            else:
                ri = rand_score(true_labels_int, pred_labels)
                ari = adjusted_rand_score(true_labels_int, pred_labels)
                nmi = normalized_mutual_info_score(true_labels_int, pred_labels)

                scatter = ax.scatter(dr_data[:, 0], dr_data[:, 1], c=pred_labels, cmap=cmap, alpha=0.7, s=15)
                ax.set_title(f'{method_name} ({dr_name})\nRI:{ri:.2f} ARI:{ari:.2f} NMI:{nmi:.2f}', fontsize=14)
                
                if n_clusters <= 15:
                     try: # legend_elements가 비어있을 수 있음
                         legend_elements = scatter.legend_elements(num=n_clusters)
                         ax.legend(legend_elements[0], [f'Cluster {k}' for k in range(n_clusters)], title="Clusters", fontsize=8, title_fontsize=10)
                     except ValueError:
                         pass # 범례 생성 실패 시 건너뛰기

            ax.set_xlabel(f'{dr_name} Dimension 1', fontsize=10)
            ax.set_ylabel(f'{dr_name} Dimension 2', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            col_idx += 1 # 다음 열로 이동

        # Ground Truth 시각화
        ax_gt = axes[i, col_idx] # 마지막 열
        scatter_gt = ax_gt.scatter(dr_data[:, 0], dr_data[:, 1], c=true_labels_int, cmap=cmap, alpha=0.7, s=15)
        ax_gt.set_title(f'Ground Truth ({dr_name})', fontsize=14)
        ax_gt.set_xlabel(f'{dr_name} Dimension 1', fontsize=10)
        ax_gt.set_ylabel(f'{dr_name} Dimension 2', fontsize=10)
        
        if n_clusters <= 15:
            try:
                legend_elements_gt = scatter_gt.legend_elements(num=n_clusters)
                # unique_labels의 길이가 n_clusters보다 짧을 수 있으므로 슬라이싱
                ax_gt.legend(legend_elements_gt[0], list(unique_labels)[:n_clusters], title="Classes", fontsize=8, title_fontsize=10)
            except ValueError:
                 pass # 범례 생성 실패 시 건너뛰기
            
        ax_gt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    # 결과 이미지 저장
    result_img_path = os.path.join(results_dir, f'{args.dataset_name}_clustering_comparison_detailed.png')
    plt.savefig(result_img_path, dpi=300, bbox_inches='tight') 
    print(f"\n✅ Clustering plot saved to {result_img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering and Visualization Script for Labeled Datasets")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name to perform clustering on (e.g., BasicMotions)')
    
    args = parser.parse_args()
    main(args)