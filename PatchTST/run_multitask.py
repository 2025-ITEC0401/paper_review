cat > run_task_specific.sh << 'EOF'
#!/bin/bash

ROOT_PATH="/hdd/dataset/newDataset"
PYTHON="python"
SCRIPT="run_multitask.py"

# 함수 정의 (실행 명령어 단축용)
run_exp() {
    DATA=$1
    TASK=$2
    SEQ_LEN=$3
    ENC_IN=$4
    BATCH=$5
    
    echo "========================================================"
    echo "Dataset: $DATA | Task: $TASK"
    echo "========================================================"
    
    $PYTHON $SCRIPT \
      --dataset_name $DATA \
      --root_path $ROOT_PATH \
      --task_name $TASK \
      --seq_len $SEQ_LEN \
      --enc_in $ENC_IN \
      --batch_size $BATCH \
      --patch_len 16 --stride 8 --epochs 10 --gpu 0
}

# ========================================================
# 1. AtrialFibrillation (Length: 640, Ch: 2)
# Tasks: Anomaly Detection, Classification, Forecasting
# ========================================================
run_exp AtrialFibrillation anomaly_detection 640 2 16
run_exp AtrialFibrillation classification 640 2 16
run_exp AtrialFibrillation forecasting 640 2 16


# ========================================================
# 2. PEMS-SF (Length: 963, Ch: 963)
# Tasks: Anomaly Detection, Forecasting, Imputation
# *Note: 채널이 많으므로 batch_size=4로 축소
# ========================================================
run_exp PEMS-SF anomaly_detection 963 963 4
run_exp PEMS-SF forecasting 963 963 4
run_exp PEMS-SF imputation 963 963 4


# ========================================================
# 3. StandWalkJump (Length: 2500, Ch: 4)
# Tasks: Anomaly Detection, Classification, Clustering
# *Clustering은 Representation 학습(Forecasting 기반)으로 수행
# ========================================================
run_exp StandWalkJump anomaly_detection 2500 4 16
run_exp StandWalkJump classification 2500 4 16
run_exp StandWalkJump clustering 2500 4 16


# ========================================================
# 4. ArticularyWordRecognition (Length: 144, Ch: 9)
# Tasks: Classification
# ========================================================
run_exp ArticularyWordRecognition classification 144 9 16


# ========================================================
# 5. NATOPS (Length: 51, Ch: 24)
# Tasks: Classification
# ========================================================
run_exp NATOPS classification 51 24 16


# ========================================================
# 6. PenDigits (Length: 8, Ch: 2)
# Tasks: Classification, Clustering
# *길이가 짧아 stride 조정은 파이썬 내부에서 처리되거나 에러날 수 있으니 주의
# ========================================================
run_exp PenDigits classification 8 2 16
run_exp PenDigits clustering 8 2 16


# ========================================================
# 7. UWaveGestureLibrary (Length: 315, Ch: 3)
# Tasks: Classification
# ========================================================
run_exp UWaveGestureLibrary classification 315 3 16

echo " 모든 Task별 실험이 완료되었습니다."
EOF