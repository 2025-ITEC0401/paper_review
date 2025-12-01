#!/bin/bash
# Merit 전체 데이터셋 학습 스크립트
# GPU 0, 1을 병렬로 활용하여 최단 시간에 완료

PYTHON="/hdd/conda_envs/envs/MERIT/bin/python3"
ROOT_PATH="/hdd/dataset/newDataset"
LLM_PATH="/hdd/intern/Merit/llm-models/llama-3.1-8b-instruct"
WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# logs 폴더 생성
mkdir -p "${WORK_DIR}/logs"

cd "${WORK_DIR}"

echo "=========================================="
echo "Merit Training Pipeline Started"
echo "Start Time: $(date)"
echo "=========================================="

# GPU 0: ArticularyWordRecognition(20M) -> PenDigits(2.1M) -> AtrialFibrillation(1.1M)
(
    echo "[GPU 0] Starting ArticularyWordRecognition..."
    $PYTHON -m src.train \
        --dataset_name ArticularyWordRecognition \
        --root_path $ROOT_PATH \
        --llm_path $LLM_PATH \
        --gpu 0 \
        --epochs 10 \
        --lr 1e-3 \
        --weight_decay 5e-4 \
        --bank_size 20 \
        > logs/ArticularyWordRecognition.log 2>&1
    echo "[GPU 0] ArticularyWordRecognition done. $(date)"

    echo "[GPU 0] Starting PenDigits..."
    $PYTHON -m src.train \
        --dataset_name PenDigits \
        --root_path $ROOT_PATH \
        --llm_path $LLM_PATH \
        --gpu 0 \
        --epochs 10 \
        --lr 1e-3 \
        --weight_decay 5e-4 \
        --bank_size 20 \
        > logs/PenDigits.log 2>&1
    echo "[GPU 0] PenDigits done. $(date)"

    echo "[GPU 0] Starting AtrialFibrillation..."
    $PYTHON -m src.train \
        --dataset_name AtrialFibrillation \
        --root_path $ROOT_PATH \
        --llm_path $LLM_PATH \
        --gpu 0 \
        --epochs 10 \
        --lr 1e-3 \
        --weight_decay 5e-4 \
        --bank_size 20 \
        > logs/AtrialFibrillation.log 2>&1
    echo "[GPU 0] AtrialFibrillation done. $(date)"
    echo "[GPU 0] All tasks completed!"
) &
GPU0_PID=$!

# GPU 1: UWaveGestureLibrary(9.8M) -> NATOPS(8.2M) -> StandWalkJump(5.0M)
(
    echo "[GPU 1] Starting UWaveGestureLibrary..."
    $PYTHON -m src.train \
        --dataset_name UWaveGestureLibrary \
        --root_path $ROOT_PATH \
        --llm_path $LLM_PATH \
        --gpu 1 \
        --epochs 10 \
        --lr 1e-3 \
        --weight_decay 5e-4 \
        --bank_size 20 \
        > logs/UWaveGestureLibrary.log 2>&1
    echo "[GPU 1] UWaveGestureLibrary done. $(date)"

    echo "[GPU 1] Starting NATOPS..."
    $PYTHON -m src.train \
        --dataset_name NATOPS \
        --root_path $ROOT_PATH \
        --llm_path $LLM_PATH \
        --gpu 1 \
        --epochs 10 \
        --lr 1e-3 \
        --weight_decay 5e-4 \
        --bank_size 20 \
        > logs/NATOPS.log 2>&1
    echo "[GPU 1] NATOPS done. $(date)"

    echo "[GPU 1] Starting StandWalkJump..."
    $PYTHON -m src.train \
        --dataset_name StandWalkJump \
        --root_path $ROOT_PATH \
        --llm_path $LLM_PATH \
        --gpu 1 \
        --epochs 10 \
        --lr 1e-3 \
        --weight_decay 5e-4 \
        --bank_size 20 \
        > logs/StandWalkJump.log 2>&1
    echo "[GPU 1] StandWalkJump done. $(date)"
    echo "[GPU 1] All tasks completed!"
) &
GPU1_PID=$!

echo "GPU 0 PID: $GPU0_PID"
echo "GPU 1 PID: $GPU1_PID"
echo ""
echo "Training running in background..."
echo "Check logs in: ${WORK_DIR}/logs/"
echo ""
echo "To monitor progress:"
echo "  tail -f logs/*.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep src.train"
