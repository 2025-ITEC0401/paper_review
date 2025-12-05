#!/bin/bash

# 설정
DATASET="PEMS-SF"
ROOT_PATH="/hdd/dataset/newDataset"

# OOM 방지를 위한 메모리 할당 정책 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "========================================================"
echo " Starting Imputation for $DATASET (Low Memory Mode)..."
echo "========================================================"

python run_imputation.py \
  --dataset_name $DATASET \
  --root_path $ROOT_PATH \
  --mask_ratio 0.25 \
  --batch_size 1 \
  --epochs 10 \
  --gpu 0 \
  --d_model 64 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 128 \
  --patch_len 16 \
  --stride 8

echo "Done."
