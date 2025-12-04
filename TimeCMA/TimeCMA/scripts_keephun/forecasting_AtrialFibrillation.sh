#!/bin/bash

# ==========================================
# 1. 환경 설정
# ==========================================
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1  # 사용할 GPU 번호

# 로그 파일 이름
LOG_FILE="forecast_AtrialFibrillation.log"

echo "=================================================="
echo "Start Forecasting (Ridge Regression)"
echo "Date: $(date)"
echo "Output Log: $LOG_FILE"
echo "=================================================="

# ==========================================
# 2. 실행 명령어 (nohup 적용)
# ==========================================
# AtrialFibrillation_m은 보통 분류용이지만, Forecasting 코드를 돌리면 
# "다음 96 스텝의 값"을 예측하는 회귀 문제로 풉니다.
# num_nodes=3 (AF 데이터셋 노드 수에 맞춤)

nohup /hdd/conda_envs/envs/TimeCMA/bin/python downstream_forecasting.py \
  --data_path AtrialFibrillation_m \
  --root_path "/hdd/intern/keephun/TimeCMA/sfs-common/dataset/" \
  --batch_size 64 \
  --num_nodes 3 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.7 \
  --e_layer 1 \
  --d_layer 2 \
  --checkpoint "/hdd/intern/keephun/TimeCMA/logs/2025-12-02-22:17:52-/AtrialFibrillation_m/96_64_1_2_0.0001_0.7_2024/best_model.pth" \
  --target_col OT \
  --feature_type latent \
  --output_dir "./Results/forecasting/" \
  --device cuda:1 > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "To check logs: tail -f $LOG_FILE"