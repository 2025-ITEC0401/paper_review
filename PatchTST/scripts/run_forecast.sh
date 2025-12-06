#!/bin/bash
ROOT_PATH="/hdd/dataset/newDataset"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 1. AtrialFibrillation (Standard)
echo "------------------------------------------------"
echo "Running Forecast: AtrialFibrillation"
echo "------------------------------------------------"
python run_forecasting.py \
  --dataset_name AtrialFibrillation \
  --root_path $ROOT_PATH \
  --pred_len 96 \
  --batch_size 16 \
  --epochs 10 \
  --gpu 0

# 2. PEMS-SF (High Channel -> Low Memory Config)
echo "------------------------------------------------"
echo "Running Forecast: PEMS-SF"
echo "------------------------------------------------"
python run_forecasting.py \
  --dataset_name PEMS-SF \
  --root_path $ROOT_PATH \
  --pred_len 24 \
  --batch_size 1 \
  --d_model 64 \
  --n_heads 4 \
  --e_layers 2 \
  --gpu 0

echo "Done."
