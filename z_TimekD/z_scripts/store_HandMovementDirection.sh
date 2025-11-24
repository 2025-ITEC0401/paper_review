#!/bin/bash
#export PYTHONPATH=/home/intern/.local/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=/hdd/conda_envs/envs/timeKD/lib/python3.10/site-packages:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=8

data_paths=("HandMovementDirection")
divides=("train" "val") 
device="cuda:1"
num_nodes=10
input_len=96
#output_len_values=(24 36 48 96 192)
output_len_values=(24)
model_name=("gpt2")
d_model=768
l_layer=12

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    for output_len in "${output_len_values[@]}"; do
      log_file="./logs/${data_path}_${output_len}_${divide}.log"
      /hdd/conda_envs/envs/timeKD/bin/python store_emb.py \
        --data_path $data_path \
        --divide $divide \
        --device $device \
        --num_nodes $num_nodes \
        --input_len $input_len \
        --output_len $output_len \
        --model_name $model_name \
        --d_model $d_model \
        --l_layer $l_layer #> $log_file
    done
  done
done
