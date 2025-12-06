#!/bin/bash

# 사용할 데이터셋의 루트 경로
ROOT_PATH="/hdd/dataset/newDataset"

# 1. BasicMotions 평가
echo "--- 1. 평가 시작: BasicMotions (GPU 1) ---"
/hdd/conda_envs/envs/patchtst/bin/python evaluate_clusters.py \
  --dataset_name BasicMotions \
  --root_path $ROOT_PATH \
  --seq_len 100 \
  --enc_in 6 \
  --patch_len 10 \
  --stride 5 \
  --gpu 1

# 2. Epilepsy 평가
echo "--- 2. 평가 시작: Epilepsy (GPU 1) ---"
/hdd/conda_envs/envs/patchtst/bin/python evaluate_clusters.py \
  --dataset_name Epilepsy \
  --root_path $ROOT_PATH \
  --seq_len 206 \
  --enc_in 3 \
  --patch_len 16 \
  --stride 8 \
  --gpu 1

# 3. Libras 평가
echo "--- 3. 평가 시작: Libras (GPU 1) ---"
/hdd/conda_envs/envs/patchtst/bin/python evaluate_clusters.py \
  --dataset_name Libras \
  --root_path $ROOT_PATH \
  --seq_len 45 \
  --enc_in 2 \
  --patch_len 8 \
  --stride 4 \
  --gpu 1

# 4. HandMovementDirection 평가
echo "--- 4. 평가 시작: HandMovementDirection (GPU 1) ---"
/hdd/conda_envs/envs/patchtst/bin/python evaluate_clusters.py \
  --dataset_name HandMovementDirection \
  --root_path $ROOT_PATH \
  --seq_len 400 \
  --enc_in 10 \
  --patch_len 16 \
  --stride 8 \
  --gpu 1

echo "--- 모든 평가 및 시각화 완료 ---"
