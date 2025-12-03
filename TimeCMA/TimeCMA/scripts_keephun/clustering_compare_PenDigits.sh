#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

# Dataset configuration
data_path="PenDigits_m"
seq_len=96
batch_size=64
num_nodes=3
n_classes=15

# Model configuration
channel=64
e_layer=1
d_layer=2
dropout_n=0.7

# Path to the trained model checkpoint
checkpoint_path="/hdd/intern/keephun/TimeCMA/logs/2025-12-02-16:50:41-/PenDigits_m/96_64_1_2_0.0001_0.7_2024/best_model.pth"

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
echo "1. K-Means Clustering (k=$n_classes)"
log_file="${output_dir}kmeans_k${n_classes}/clustering.log"
mkdir -p "${output_dir}kmeans_k${n_classes}/"
/hdd/conda_envs/envs/TimeCMA/bin/python downstream.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes $num_nodes \
  --seq_len $seq_len \
  --pred_len 96 \
  --channel $channel \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --checkpoint $checkpoint_path \
  --clustering_method kmeans \
  --n_classes $n_classes \
  --feature_type $feature_type \
  --output_dir "${output_dir}kmeans_k${n_classes}/" \
  --device cuda:1
echo "  - PID: $! (log: $log_file)"

echo ""
echo "2. Spectral Clustering (k=$n_classes)..."
log_file="${output_dir}spectral_k${n_classes}/clustering.log"
mkdir -p "${output_dir}spectral_k${n_classes}/"
/hdd/conda_envs/envs/TimeCMA/bin/python downstream.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes $num_nodes \
  --seq_len $seq_len \
  --pred_len 96 \
  --channel $channel \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --checkpoint $checkpoint_path \
  --clustering_method spectral \
  --n_classes $n_classes \
  --feature_type $feature_type \
  --output_dir "${output_dir}spectral_k${n_classes}/" \
  --device cuda:1

echo "  - Log: $log_file"


echo ""
echo "=================================================="
echo "Clustering Comparison Completed!"
echo "Results saved to: $output_dir"
echo "=================================================="
echo ""
echo "Summary of results:"
echo "-------------------"

for method_dir in ${output_dir}*/; do
  result_file=$(ls ${method_dir}${data_path}_*_${feature_type}_results.txt 2>/dev/null | head -n 1)
  if [ -f "$result_file" ]; then
    echo ""
    echo "$(basename $method_dir):"
    grep -E "Rand Index|Normalized Mutual Information|Number of clusters" "$result_file"
  fi
done