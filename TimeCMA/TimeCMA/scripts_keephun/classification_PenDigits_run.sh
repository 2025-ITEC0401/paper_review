#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

echo "=================================================="
echo "Start Classification: AtrialFibrillation_m (SVM)"
echo "Date: $(date)"
echo "=================================================="
/hdd/conda_envs/envs/TimeCMA/bin/python downstream_classification.py \
  --data_path PenDigits_m \
  --root_path "/hdd/intern/keephun/TimeCMA/sfs-common/dataset/" \
  --batch_size 64 \
  --num_nodes 3 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 64 \
  --dropout_n 0.7 \
  --e_layer 1 \
  --d_layer 2 \
  --checkpoint "/hdd/intern/keephun/TimeCMA/logs/2025-12-02-16:50:41-/PenDigits_m/96_64_1_2_0.0001_0.7_2024/best_model.pth" \
  --classifier svm \
  --target_col OT \
  --feature_type latent \
  --output_dir "./Results/classification_comparison/PenDigits_m/" \
  --device cuda:1

echo "=================================================="
echo "Process Finished."
echo "=================================================="