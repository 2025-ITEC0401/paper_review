cat > run_all.sh << 'EOF'
#!/bin/bash

# 데이터셋 루트 경로
ROOT_PATH="/hdd/dataset/newDataset"
# Python 실행 경로
PYTHON_PATH="python"

# --- 1. AtrialFibrillation ---
echo "--- 1. 훈련 시작: AtrialFibrillation (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name AtrialFibrillation \
  --root_path $ROOT_PATH \
  --seq_len 640 \
  --enc_in 2 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --gpu 1

# --- 2. StandWalkJump ---
echo "--- 2. 훈련 시작: StandWalkJump (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name StandWalkJump \
  --root_path $ROOT_PATH \
  --seq_len 2500 \
  --enc_in 4 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --gpu 1

# --- 3. ArticularyWordRecognition ---
echo "--- 3. 훈련 시작: ArticularyWordRecognition (GPU 1) ---"
$PYTHON_PATH run_ucr_exp.py \
  --dataset_name ArticularyWordRecognition \
  --root_path $ROOT_PATH \
  --seq_len 144 \
  --enc_in 9 \
  --patch_len 16 \
  --stride 8 \
  --epochs 20 \
  --gpu 1

# --- 4. NATOPS ---
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

# --- 5. PenDigits ---
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

# --- 7. PEMS-SF ---
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
EOF