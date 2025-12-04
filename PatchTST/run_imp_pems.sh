#!/bin/bash

# 설정
DATASET="PEMS-SF"
ROOT_PATH="/hdd/dataset/newDataset"
SEQ_LEN=963
ENC_IN=963
BATCH_SIZE=4  # 메모리 부족 방지
MASK_RATIO=0.25 # 25% 가림

echo "========================================================"
echo " Starting Imputation for $DATASET..."
echo "========================================================"

python run_imputation.py \
  --dataset_name $DATASET \
  --root_path $ROOT_PATH \
  --seq_len $SEQ_LEN \
  --enc_in $ENC_IN \
  --mask_ratio $MASK_RATIO \
  --batch_size $BATCH_SIZE \
  --epochs 10 \
  --gpu 0 \
  --d_model 128 \
  --n_heads 16 \
  --e_layers 3

echo "Done."
