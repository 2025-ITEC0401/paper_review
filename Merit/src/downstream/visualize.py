# src/downstream/visualize.py
# 다운스트림 태스크 시각화 모듈

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def visualize_representations(representations, labels, dataset_name, output_dir, method='tsne'):
    """Representation 시각화 (t-SNE 또는 PCA)"""
    plt.figure(figsize=(10, 8))

    perplexity = min(30, len(representations) - 1)
    if method == 'tsne' and perplexity >= 5:
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        reducer = PCA(n_components=2)
        method = 'pca'

    embeddings_2d = reducer.fit_transform(representations)

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_labels), 10)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            color_idx = idx % 10
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors[color_idx]], label=str(label), alpha=0.7, s=30)

        if len(unique_labels) <= 10:
            plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=30)

    plt.title(f'{dataset_name} - {method.upper()} Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'{dataset_name}_{method}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def visualize_clustering(representations, cluster_labels, true_labels, dataset_name, output_dir):
    """클러스터링 결과 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    perplexity = min(30, len(representations) - 1)
    if perplexity >= 5:
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        reducer = PCA(n_components=2)

    embeddings_2d = reducer.fit_transform(representations)

    # True labels
    ax1 = axes[0]
    unique_labels = np.unique(true_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_labels), 10)))
    for idx, label in enumerate(unique_labels):
        mask = true_labels == label
        color_idx = idx % 10
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[color_idx]], label=str(label), alpha=0.7, s=30)
    ax1.set_title(f'{dataset_name} - True Labels')
    if len(unique_labels) <= 10:
        ax1.legend(title='Class')

    # Cluster labels
    ax2 = axes[1]
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_clusters), 10)))
    for idx, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        color_idx = idx % 10
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[color_idx]], label=f'Cluster {cluster}', alpha=0.7, s=30)
    ax2.set_title(f'{dataset_name} - K-Means Clusters')
    if len(unique_clusters) <= 10:
        ax2.legend(title='Cluster')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{dataset_name}_clustering.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def create_summary_figure(all_results, output_dir):
    """전체 결과 요약 그래프"""
    datasets = list(all_results.keys())

    # Classification Accuracy
    clf_data = [(d, all_results[d]['classification']['accuracy'])
                for d in datasets if 'classification' in all_results[d]]

    if clf_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        names, accs = zip(*clf_data)
        bars = ax.bar(names, accs, color='steelblue', alpha=0.8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Accuracy by Dataset')
        ax.set_ylim(0, 1)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'classification_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Clustering NMI/ARI
    clust_data = [(d, all_results[d]['clustering']['nmi'], all_results[d]['clustering']['ari'])
                  for d in datasets if 'clustering' in all_results[d]]

    if clust_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        names, nmis, aris = zip(*clust_data)
        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, nmis, width, label='NMI', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, aris, width, label='ARI', color='coral', alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Clustering Metrics by Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'clustering_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
