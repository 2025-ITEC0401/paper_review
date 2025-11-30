#!/bin/bash

# 데이터셋 루트 경로
ROOT_PATH="/hdd/dataset/newDataset"
# Python 실행 경로 (사용자 환경)
PYTHON_PATH="/hdd/conda_envs/envs/patchtst/bin/python"

# --- 1. ArticularyWordRecognition ---
# 길이: 144, 채널: 9
echo "--- 1. 훈련 시작: ArticularyWordRecognition (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name ArticularyWordRecognition \
  --root_path $ROOT_PATH \
  --seq_len 144 \
  --enc_in 9 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --gpu 1

# --- 2. AtrialFibrillation ---
# 길이: 640, 채널: 2
echo "--- 2. 훈련 시작: AtrialFibrillation (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name AtrialFibrillation \
  --root_path $ROOT_PATH \
  --seq_len 640 \
  --enc_in 2 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --gpu 1

# --- 3. StandWalkJump ---
# 길이: 2500, 채널: 4
echo "--- 3. 훈련 시작: StandWalkJump (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name StandWalkJump \
  --root_path $ROOT_PATH \
  --seq_len 2500 \
  --enc_in 4 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --gpu 1

# --- 4. NATOPS ---
# 길이: 51, 채널: 24
echo "--- 4. 훈련 시작: NATOPS (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name NATOPS \
  --root_path $ROOT_PATH \
  --seq_len 51 \
  --enc_in 24 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --gpu 1

# --- 5. PenDigits (주의: 길이가 매우 짧음) ---
# 길이: 8, 채널: 2
# *중요*: 시퀀스 길이가 8이므로 patch_len을 4로 줄임
echo "--- 5. 훈련 시작: PenDigits (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name PenDigits \
  --root_path $ROOT_PATH \
  --seq_len 8 \
  --enc_in 2 \
  --patch_len 4 \
  --stride 2 \
  --epochs 20 \
  --gpu 1

# --- 6. UWaveGestureLibrary ---
# 길이: 315, 채널: 3
echo "--- 6. 훈련 시작: UWaveGestureLibrary (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name UWaveGestureLibrary \
  --root_path $ROOT_PATH \
  --seq_len 315 \
  --enc_in 3 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --gpu 1

# --- 7. PEMS-SF (주의: 채널이 매우 많음) ---
# 길이: 963, 채널: 963
# *중요*: 채널이 963개라 OOM 방지를 위해 batch_size를 4로 줄임
echo "--- 7. 훈련 시작: PEMS-SF (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name PEMS-SF \
  --root_path $ROOT_PATH \
  --seq_len 963 \
  --enc_in 963 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --batch_size 4 \
  --gpu 1

echo "--- 모든 새로운 데이터셋 훈련 완료 ---"