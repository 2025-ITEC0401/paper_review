import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.TimeCMA import Dual
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--channel", type=int, default=64, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=192, help="out_len")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--dropout_n", type=float, default=0.7, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=2, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    
    # Clustering specific arguments
    parser.add_argument("--n_clusters", type=int, default=5, help="number of clusters for KMeans")
    parser.add_argument("--clustering_method", type=str, default="kmeans", 
                       choices=["kmeans", "dbscan", "agglomerative"],
                       help="clustering method to use")
    parser.add_argument("--feature_type", type=str, default="latent", 
                       choices=["latent", "embedding", "raw"],
                       help="type of features to use for clustering")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./Results/clustering/", 
                       help="output directory for clustering results")
    parser.add_argument("--visualize", default=True, action="store_true", help="visualize clustering results")
    
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_features(model, data_loader, device, feature_type="latent"):
    """
    Extract features from the model for clustering
    Args:
        model: trained TimeCMA model
        data_loader: DataLoader for the dataset
        device: torch device
        feature_type: type of features to extract
            - "latent": features from cross-modal alignment layer
            - "embedding": LLM embeddings
            - "raw": normalized input data
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, embeddings in data_loader:
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            embeddings = embeddings.float().to(device)
            
            if feature_type == "raw":
                # Use normalized raw input
                normalized_data = model.normalize_layers(batch_x, 'norm')
                features = normalized_data.reshape(normalized_data.shape[0], -1)  # [B, L*N]
                
            elif feature_type == "embedding":
                # Use LLM embeddings
                embeddings_squeezed = embeddings.squeeze(-1)  # [B, E, N]
                features = embeddings_squeezed.reshape(embeddings_squeezed.shape[0], -1)  # [B, E*N]
                
            elif feature_type == "latent":
                # Extract latent features from cross-modal alignment
                # Forward pass through encoder and cross-modal alignment
                input_data = model.normalize_layers(batch_x, 'norm')
                input_data = input_data.permute(0, 2, 1)  # [B, N, L]
                input_data = model.length_to_feature(input_data)  # [B, N, C]
                
                embeddings_squeezed = embeddings.squeeze(-1)  # [B, E, N]
                embeddings_squeezed = embeddings_squeezed.permute(0, 2, 1)  # [B, N, E]
                
                # Encoder
                enc_out = model.ts_encoder(input_data)  # [B, N, C]
                enc_out = enc_out.permute(0, 2, 1)  # [B, C, N]
                prompt_enc = model.prompt_encoder(embeddings_squeezed)  # [B, N, E]
                prompt_enc = prompt_enc.permute(0, 2, 1)  # [B, E, N]
                
                # Cross-modal features
                cross_out = model.cross(enc_out, prompt_enc, prompt_enc)  # [B, C, N]
                features = cross_out.reshape(cross_out.shape[0], -1)  # [B, C*N]
            
            features_list.append(features.cpu().numpy())
            labels_list.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    return all_features, all_labels


def perform_clustering(features, method="kmeans", n_clusters=5):
    """
    Perform clustering on the extracted features
    """
    if method == "kmeans":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clustering_model.fit_predict(features)
        
    elif method == "dbscan":
        clustering_model = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clustering_model.fit_predict(features)
        
    elif method == "agglomerative":
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering_model.fit_predict(features)
    
    return cluster_labels, clustering_model


def evaluate_clustering(features, cluster_labels):
    """
    Evaluate clustering quality using various metrics
    """
    # Filter out noise points (label -1) for DBSCAN
    valid_mask = cluster_labels != -1
    features_valid = features[valid_mask]
    labels_valid = cluster_labels[valid_mask]
    
    if len(np.unique(labels_valid)) < 2:
        print("Warning: Only one cluster found. Cannot compute metrics.")
        return {}
    
    metrics = {}
    
    try:
        metrics['silhouette_score'] = silhouette_score(features_valid, labels_valid)
    except:
        metrics['silhouette_score'] = None
        
    try:
        metrics['davies_bouldin_score'] = davies_bouldin_score(features_valid, labels_valid)
    except:
        metrics['davies_bouldin_score'] = None
        
    try:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_valid, labels_valid)
    except:
        metrics['calinski_harabasz_score'] = None
    
    metrics['n_clusters'] = len(np.unique(labels_valid))
    metrics['n_samples'] = len(features_valid)
    
    if valid_mask.sum() < len(cluster_labels):
        metrics['n_noise'] = len(cluster_labels) - valid_mask.sum()
    
    return metrics


def visualize_clusters(features, cluster_labels, output_path, method="tsne"):
    """
    Visualize clustering results using dimensionality reduction
    """
    plt.figure(figsize=(12, 8))
    
    # Dimensionality reduction
    if method == "tsne":
        if features.shape[0] > 5000:
            # Subsample for faster computation
            indices = np.random.choice(features.shape[0], 5000, replace=False)
            features_reduced = features[indices]
            labels_reduced = cluster_labels[indices]
        else:
            features_reduced = features
            labels_reduced = cluster_labels
            
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = reducer.fit_transform(features_reduced)
        
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features)
        labels_reduced = cluster_labels
    
    # Plot
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_reduced, cmap='viridis', 
                         alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Clustering Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = Dual(
        device=device,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head
    )
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the checkpoint is the state_dict itself
                model.load_state_dict(checkpoint)
        else:
            # Checkpoint is directly the state_dict
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    model.to(device)
    model.eval()
    
    # Prepare data
    print("Loading data...")
    if args.data_path.startswith("ETTh"):
        test_dataset = Dataset_ETT_hour(
            flag='test',
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            num_nodes=args.num_nodes,
            model_name=args.model_name
        )
    elif args.data_path.startswith("ETTm"):
        test_dataset = Dataset_ETT_minute(
            flag='test',
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            model_name=args.model_name
        )
    else:
        test_dataset = Dataset_Custom(
            flag='test',
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            model_name=args.model_name
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    # Extract features
    print(f"Extracting {args.feature_type} features...")
    features, labels = extract_features(model, test_loader, device, args.feature_type)
    print(f"Extracted features shape: {features.shape}")
    
    # Perform clustering
    print(f"Performing {args.clustering_method} clustering...")
    cluster_labels, clustering_model = perform_clustering(
        features, 
        method=args.clustering_method, 
        n_clusters=args.n_clusters
    )
    
    # Evaluate clustering
    print("Evaluating clustering quality...")
    metrics = evaluate_clustering(features, cluster_labels)
    
    # Print results
    print("\n" + "="*50)
    print("Clustering Results")
    print("="*50)
    print(f"Method: {args.clustering_method}")
    print(f"Feature type: {args.feature_type}")
    print(f"Number of clusters: {metrics.get('n_clusters', 'N/A')}")
    print(f"Number of samples: {metrics.get('n_samples', 'N/A')}")
    if 'n_noise' in metrics:
        print(f"Number of noise points: {metrics['n_noise']}")
    print(f"\nQuality Metrics:")
    print(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}" if metrics.get('silhouette_score') else "  Silhouette Score: N/A")
    print(f"  Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 'N/A'):.4f}" if metrics.get('davies_bouldin_score') else "  Davies-Bouldin Score: N/A")
    print(f"  Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 'N/A'):.4f}" if metrics.get('calinski_harabasz_score') else "  Calinski-Harabasz Score: N/A")
    print("="*50)
    
    # Save results
    result_file = os.path.join(
        args.output_dir,
        f"{args.data_path}_{args.clustering_method}_{args.feature_type}_results.txt"
    )
    with open(result_file, 'w') as f:
        f.write("Clustering Results\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Method: {args.clustering_method}\n")
        f.write(f"Feature type: {args.feature_type}\n")
        f.write(f"Sequence length: {args.seq_len}\n")
        f.write(f"Prediction length: {args.pred_len}\n")
        f.write(f"Number of clusters: {metrics.get('n_clusters', 'N/A')}\n")
        f.write(f"Number of samples: {metrics.get('n_samples', 'N/A')}\n")
        if 'n_noise' in metrics:
            f.write(f"Number of noise points: {metrics['n_noise']}\n")
        f.write(f"\nQuality Metrics:\n")
        f.write(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A')}\n")
        f.write(f"  Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 'N/A')}\n")
        f.write(f"  Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 'N/A')}\n")
    
    print(f"\nResults saved to {result_file}")
    
    # Save cluster labels
    labels_file = os.path.join(
        args.output_dir,
        f"{args.data_path}_{args.clustering_method}_{args.feature_type}_labels.npy"
    )
    np.save(labels_file, cluster_labels)
    print(f"Cluster labels saved to {labels_file}")
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # t-SNE visualization
        vis_path_tsne = os.path.join(
            args.output_dir,
            f"{args.data_path}_{args.clustering_method}_{args.feature_type}_tsne.png"
        )
        visualize_clusters(features, cluster_labels, vis_path_tsne, method="tsne")
        
        # PCA visualization
        vis_path_pca = os.path.join(
            args.output_dir,
            f"{args.data_path}_{args.clustering_method}_{args.feature_type}_pca.png"
        )
        visualize_clusters(features, cluster_labels, vis_path_pca, method="pca")
        
        # Cluster size distribution
        plt.figure(figsize=(10, 6))
        unique, counts = np.unique(cluster_labels, return_counts=True)
        plt.bar(unique, counts)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Samples')
        plt.title('Cluster Size Distribution')
        plt.tight_layout()
        dist_path = os.path.join(
            args.output_dir,
            f"{args.data_path}_{args.clustering_method}_{args.feature_type}_distribution.png"
        )
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribution plot saved to {dist_path}")
    
    print("\nClustering analysis completed!")


if __name__ == "__main__":
    main()
