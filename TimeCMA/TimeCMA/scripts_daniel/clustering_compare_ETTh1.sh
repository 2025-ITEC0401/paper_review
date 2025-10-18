#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Dataset configuration
data_path="ETTh1"
seq_len=96
batch_size=16
num_nodes=7

# Model configuration
channel=64
e_layer=1
d_layer=2
dropout_n=0.7

# Path to the trained model checkpoint
checkpoint_path="./logs/2025-10-17-03:08:33-/ETTh1/96_64_1_2_0.0001_0.7_2024/best_model.pth"

# Output directory
output_dir="./Results/clustering_comparison/${data_path}/"
mkdir -p $output_dir

echo "=================================================="
echo "Clustering Method Comparison"
echo "=================================================="
echo "Dataset: $data_path"
echo "Checkpoint: $checkpoint_path"
echo "Testing all clustering methods with latent features"
echo "=================================================="

# Test different clustering methods with latent features
feature_type="latent"

echo ""
echo "1. K-Means Clustering (k=3,5,7)..."
for n_clusters in 3 5 7; do
  echo "  - Running with k=$n_clusters in background..."
  log_file="${output_dir}kmeans_k${n_clusters}/clustering.log"
  mkdir -p "${output_dir}kmeans_k${n_clusters}/"
  nohup /hdd/conda_envs/envs/TimeCMA/bin/python downstream.py \
    --data_path $data_path \
    --batch_size $batch_size \
    --num_nodes 7 \
    --seq_len $seq_len \
    --pred_len 96 \
    --channel $channel \
    --dropout_n $dropout_n \
    --e_layer $e_layer \
    --d_layer $d_layer \
    --checkpoint $checkpoint_path \
    --clustering_method kmeans \
    --n_clusters $n_clusters \
    --feature_type $feature_type \
    --output_dir "${output_dir}kmeans_k${n_clusters}/" \
    --device cuda:0 > $log_file 2>&1 &
  echo "  - PID: $! (log: $log_file)"
done

echo ""
echo "2. DBSCAN Clustering..."
log_file="${output_dir}dbscan/clustering.log"
mkdir -p "${output_dir}dbscan/"
nohup /hdd/conda_envs/envs/TimeCMA/bin/python downstream.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len 96 \
  --channel $channel \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --checkpoint $checkpoint_path \
  --clustering_method dbscan \
  --feature_type $feature_type \
  --output_dir "${output_dir}dbscan/" \
  --device cuda:0 > $log_file 2>&1 &
echo "  - PID: $! (log: $log_file)"

echo ""
echo "3. Agglomerative Clustering (k=3,5,7)..."
for n_clusters in 3 5 7; do
  echo "  - Running with k=$n_clusters in background..."
  log_file="${output_dir}agglomerative_k${n_clusters}/clustering.log"
  mkdir -p "${output_dir}agglomerative_k${n_clusters}/"
  nohup /hdd/conda_envs/envs/TimeCMA/bin/python downstream.py \
    --data_path $data_path \
    --batch_size $batch_size \
    --num_nodes 7 \
    --seq_len $seq_len \
    --pred_len 96 \
    --channel $channel \
    --dropout_n $dropout_n \
    --e_layer $e_layer \
    --d_layer $d_layer \
    --checkpoint $checkpoint_path \
    --clustering_method agglomerative \
    --n_clusters $n_clusters \
    --feature_type $feature_type \
    --output_dir "${output_dir}agglomerative_k${n_clusters}/" \
    --device cuda:0 > $log_file 2>&1 &
  echo "  - PID: $! (log: $log_file)"
done

echo ""
echo "=================================================="
echo "Clustering Comparison Completed!"
echo "Results saved to: $output_dir"
echo "=================================================="
echo ""
echo "Summary of results:"
echo "-------------------"
for method_dir in ${output_dir}*/; do
  result_file="${method_dir}${data_path}_*_${feature_type}_results.txt"
  if ls $result_file 1> /dev/null 2>&1; then
    echo ""
    echo "$(basename $method_dir):"
    grep -E "Silhouette Score|Davies-Bouldin|Calinski-Harabasz|Number of clusters" $result_file | head -4
  fi
done
