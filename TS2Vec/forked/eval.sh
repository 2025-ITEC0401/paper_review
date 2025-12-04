#!/bin/bash

# UEA 데이터셋 평가 스크립트
# 사용법: ./eval.sh <dataset_name> <model_path> [options]
# 예시: ./eval.sh BasicMotions training/BasicMotions__UEA_20251020_002028/model_best.pkl

# 기본 설정
GPU=0
EVAL_PROTOCOL="svm"
DEFAULT_TASKS=""

# 데이터셋별 TASKS 설정 함수
get_tasks_for_dataset() {
    local dataset=$1
    case $dataset in
        "AtrialFibrillation")
            echo "imputation"
            ;;
        "PEMS-SF")
            echo "imputation"
            ;;
        "StandWalkJump")
            echo "classification"
            ;;
        # 기본값
        *)
            echo "$DEFAULT_TASKS"
            ;;
    esac
}

# 태스크별 평가 실행 함수
run_eval_by_task() {
    local dataset=$1
    local model_path=$2
    local tasks=$3
    
    # tasks를 쉼표로 분리하여 배열로 변환
    IFS=',' read -ra task_array <<< "$tasks"
    
    # 태스크 분류
    local anomaly_tasks=""
    local forecast_tasks=""
    local other_tasks=""
    
    for task in "${task_array[@]}"; do
        task=$(echo "$task" | xargs)  # trim whitespace
        case $task in
            "anomaly_detection")
                anomaly_tasks="anomaly_detection"
                ;;
            "forecasting")
                forecast_tasks="forecasting"
                ;;
            *)
                if [ -z "$other_tasks" ]; then
                    other_tasks="$task"
                else
                    other_tasks="$other_tasks,$task"
                fi
                ;;
        esac
    done
    
    # anomaly_detection 태스크 실행 (UEA_anomaly 로더)
    if [ -n "$anomaly_tasks" ]; then
        echo ""
        echo "--- Running anomaly_detection with UEA_anomaly loader ---"
        /hdd/conda_envs/envs/ts2vec_daniel/bin/python eval.py $dataset \
            --model-path $model_path \
            --loader UEA_anomaly \
            --gpu $GPU \
            --eval-protocol $EVAL_PROTOCOL \
            --tasks $anomaly_tasks
    fi
    
    # forecasting 태스크 실행 (UEA_forecast 로더)
    if [ -n "$forecast_tasks" ]; then
        echo ""
        echo "--- Running forecasting with UEA_forecast loader ---"
        /hdd/conda_envs/envs/ts2vec_daniel/bin/python eval.py $dataset \
            --model-path $model_path \
            --loader UEA_forecast \
            --gpu $GPU \
            --eval-protocol $EVAL_PROTOCOL \
            --tasks $forecast_tasks
    fi
    
    # 나머지 태스크 실행 (UEA 로더)
    if [ -n "$other_tasks" ]; then
        echo ""
        echo "--- Running $other_tasks with UEA loader ---"
        /hdd/conda_envs/envs/ts2vec_daniel/bin/python eval.py $dataset \
            --model-path $model_path \
            --loader UEA \
            --gpu $GPU \
            --eval-protocol $EVAL_PROTOCOL \
            --tasks $other_tasks \
            --missing-ratios 0.2 \
            --missing-types random
    fi
}

# 단일 데이터셋 평가
if [ $# -ge 2 ]; then
    DATASET=$1
    MODEL_PATH=$2
    
    # 3번째 인자가 있으면 TASKS로 사용, 없으면 데이터셋별 기본값 사용
    if [ $# -ge 3 ]; then
        TASKS=$3
    else
        TASKS=$(get_tasks_for_dataset $DATASET)
    fi
    
    echo "=========================================="
    echo "Evaluating: $DATASET"
    echo "Model: $MODEL_PATH"
    echo "Tasks: $TASKS"
    echo "=========================================="
    
    run_eval_by_task $DATASET $MODEL_PATH "$TASKS"
    
    exit 0
fi

# 인자가 없으면 전체 UEA 데이터셋 평가 (training 폴더 내 모든 모델)
echo "=========================================="
echo "Batch Evaluation for UEA Datasets"
echo "=========================================="

# UEA 데이터셋 목록
datasets=(
    # "ArticularyWordRecognition"
    # "AtrialFibrillation"
    # "NATOPS"
    "PEMS-SF"
    # "PenDigits"
    # "StandWalkJump"
    # "UWaveGestureLibrary"
)

# 각 데이터셋에 대해 최신 모델 찾아서 평가
for dataset in "${datasets[@]}"; do
    # 해당 데이터셋의 최신 training 폴더 찾기
    LATEST_DIR=$(ls -dt result/${dataset}__UEA_* 2>/dev/null | head -1)
    
    if [ -z "$LATEST_DIR" ]; then
        echo "Skipping $dataset: No training folder found"
        continue
    fi
    
    # model_best.pkl 우선, 없으면 model.pkl 사용
    if [ -f "$LATEST_DIR/model_best.pkl" ]; then
        MODEL_PATH="$LATEST_DIR/model_best.pkl"
    elif [ -f "$LATEST_DIR/model.pkl" ]; then
        MODEL_PATH="$LATEST_DIR/model.pkl"
    else
        echo "Skipping $dataset: No model found in $LATEST_DIR"
        continue
    fi
    
    # 데이터셋별 TASKS 가져오기
    TASKS=$(get_tasks_for_dataset $dataset)
    
    echo ""
    echo "=========================================="
    echo "Evaluating: $dataset"
    echo "Model: $MODEL_PATH"
    echo "Tasks: $TASKS"
    echo "=========================================="
    
    run_eval_by_task $dataset $MODEL_PATH "$TASKS"
done

echo ""
echo "=========================================="
echo "Batch Evaluation Completed!"
echo "=========================================="
