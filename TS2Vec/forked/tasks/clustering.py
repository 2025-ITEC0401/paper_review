
import sys
if __name__ == '__main__':
    sys.path.append('/hdd/intern/daniel/paper_review/TS2Vec/forked/')

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

import datautils
from ts2vec import TS2Vec


def _clustering_accuracy(true_labels, pred_labels):
    contingency = contingency_matrix(true_labels, pred_labels)
    if contingency.size == 0:
        return 0.0
    cost = contingency.max() - contingency
    row_ind, col_ind = linear_sum_assignment(cost)
    return contingency[row_ind, col_ind].sum() / contingency.sum()

def eval_clustering(savedir, model, test_data, test_labels, random_state=0):
    test_repr = model.encode(test_data, encoding_window='full_series')

    labels = test_labels

    pca = PCA(n_components=2)
    test_repr_2d = pca.fit_transform(test_repr)

    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        raise ValueError('Need at least two clusters for clustering evaluation.')

    clusterers = {
        'kmeans': KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
        'spectral': SpectralClustering(
            n_clusters=n_clusters,
        )
    }

    predictions = {}
    metrics = {}
    for name, clusterer in clusterers.items():
        pred = clusterer.fit_predict(test_repr)
        predictions[name] = pred
        metrics[name] = {
            'acc': float(_clustering_accuracy(labels, pred)),
            'ri': float(rand_score(labels, pred)),
            'nmi': float(normalized_mutual_info_score(labels, pred))
        }

        plt.figure(figsize=(8, 6))
        for i in range(n_clusters):
            plt.scatter(test_repr_2d[pred == i, 0], test_repr_2d[pred == i, 1], label=f'Cluster {i}', s=50)
        plt.title(f'Clustering result using {name.capitalize()}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster Label')
        plt.savefig(Path(savedir) / f'clustering_{name}.png')
        plt.close()

    out = {
        'true_labels': labels.tolist(),
        'kmeans_labels': predictions['kmeans'].tolist(),
        'spectral_labels': predictions['spectral'].tolist(),
    }
    
    return out, metrics

def main():
    dataset_name = 'BasicMotions'
    model_path = '/hdd/intern/daniel/paper_review/TS2Vec/forked/training/BasicMotions__UEA_20251019_220112/model.pkl'
    device = 'cuda'

    print(f'Loading UEA dataset: {dataset_name}')
    train_data, train_labels, test_data, test_labels = datautils.load_UEA(dataset_name)

    print(f'Initializing TS2Vec model on device: {device}')
    model = TS2Vec(input_dims=train_data.shape[-1], device=device)
    print(f'Loading model weights from {model_path}')
    model.load(model_path)

    print('Running clustering evaluation (kmeans, spectral)...')
    out, metrics = eval_clustering("./", model, train_data, train_labels, test_data, test_labels)

    for method, scores in metrics.items():
        acc = scores['acc']
        ri = scores['ri']
        nmi = scores['nmi']
        print(f"{method}: acc={acc:.4f}, ri={ri:.4f}, nmi={nmi:.4f}")


if __name__ == '__main__':
    main()
