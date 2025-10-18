#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Dataset configuration
data_path="ETTh1"
seq_len=96
batch_size=256
num_nodes=7

# Model configuration
pred_len=96  # MUST match the checkpoint! (96_64_1_2 means pred_len=96)
channel=64
e_layer=1
d_layer=2
dropout_n=0.7

# Clustering configuration
n_clusters=5
clustering_method="kmeans"  # Options: kmeans, dbscan, agglomerative
feature_type="latent"        # Options: latent, embedding, raw

# Path to the trained model checkpoint
# Use the latest checkpoint
# Format: {pred_len}_{channel}_{e_layer}_{d_layer}_{lr}_{dropout_n}_{seed}
checkpoint_path="./logs/2025-10-17-03:08:33-/ETTh1/96_64_1_2_0.0001_0.7_2024/best_model.pth"

# Output directory
output_dir="./Results/clustering/${data_path}/"
mkdir -p $output_dir

echo "=================================================="
echo "Running Clustering Analysis"
echo "=================================================="
echo "Dataset: $data_path"
echo "Checkpoint: $checkpoint_path"
echo "Clustering Method: $clustering_method"
echo "Feature Type: $feature_type"
echo "Number of Clusters: $n_clusters"
echo "=================================================="

# Run clustering with latent features
echo ""
echo "1. Running clustering with LATENT features..."
/hdd/conda_envs/envs/TimeCMA/bin/python downstream.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes $num_nodes \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --channel $channel \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --checkpoint $checkpoint_path \
  --clustering_method $clustering_method \
  --n_clusters $n_clusters \
  --feature_type latent \
  --output_dir $output_dir \
  --visualize \
  --device cuda:0

echo ""
echo "2. Running clustering with EMBEDDING features..."
/hdd/conda_envs/envs/TimeCMA/bin/python downstream.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes $num_nodes \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --channel $channel \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --checkpoint $checkpoint_path \
  --clustering_method $clustering_method \
  --n_clusters $n_clusters \
  --feature_type embedding \
  --output_dir $output_dir \
  --visualize \
  --device cuda:0

echo ""
echo "3. Running clustering with RAW features..."
/hdd/conda_envs/envs/TimeCMA/bin/python downstream.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes $num_nodes \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --channel $channel \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --checkpoint $checkpoint_path \
  --clustering_method $clustering_method \
  --n_clusters $n_clusters \
  --feature_type raw \
  --output_dir $output_dir \
  --visualize \
  --device cuda:0

echo ""
echo "=================================================="
echo "Clustering Analysis Completed!"
echo "Results saved to: $output_dir"
echo "=================================================="
