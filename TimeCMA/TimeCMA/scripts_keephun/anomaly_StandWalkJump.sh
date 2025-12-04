#!/bin/bash

# ==========================================
# 1. 환경 설정
# ==========================================
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1  # 사용할 GPU 번호

# 로그 파일 이름
LOG_FILE="anomaly_StandWalkJump.log"

echo "=================================================="
echo "Start Anomaly Detection (IsoForest & OC-SVM)"
echo "Date: $(date)"
echo "Output Log: $LOG_FILE"
echo "=================================================="

# ==========================================
# 2. 실행 명령어 (nohup 적용)
# ==========================================
nohup /hdd/conda_envs/envs/TimeCMA/bin/python downstream_anomaly.py \
  --data_path StandWalkJump_m \
  --root_path "/hdd/intern/keephun/TimeCMA/sfs-common/dataset/" \
  --batch_size 64 \
  --num_nodes 5 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.7 \
  --e_layer 1 \
  --d_layer 2 \
  --checkpoint "/hdd/intern/keephun/TimeCMA/logs/2025-12-03-10:57:17-/StandWalkJump_m/96_64_1_2_0.0001_0.7_2024/best_model.pth" \
  --target_col OT \
  --feature_type latent \
  --contamination 0.1 \
  --output_dir "./Results/anomaly_models/" \
  --device cuda:0 > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "To check logs: tail -f $LOG_FILE"