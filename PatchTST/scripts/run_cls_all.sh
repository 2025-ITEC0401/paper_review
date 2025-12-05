#!/bin/bash

ROOT_PATH="/hdd/dataset/newDataset"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 1. 일반 데이터셋 (Batch Size 16)
datasets_std=(
    "ArticularyWordRecognition"
    "AtrialFibrillation"
    "NATOPS"
    "UWaveGestureLibrary"
    "PenDigits"  # 이제 자동 패치 조절로 돌아감
)

echo "=== Standard Datasets ==="
for DATASET in "${datasets_std[@]}"; do
    echo ">>> Processing: $DATASET"
    python run_classification.py \
      --dataset_name $DATASET \
      --root_path $ROOT_PATH \
      --batch_size 16 \
      --epochs 20 \
      --gpu 0
    echo "--------------------------------------------------------"
done

# 2. 고용량 데이터셋 (Batch Size 4) - OOM 방지
# StandWalkJump는 길이가 2500이라 메모리가 많이 필요함
echo "=== Heavy Dataset (StandWalkJump) ==="
echo ">>> Processing: StandWalkJump"
python run_classification.py \
  --dataset_name StandWalkJump \
  --root_path $ROOT_PATH \
  --batch_size 4 \
  --epochs 20 \
  --gpu 0

echo "All tasks completed."
