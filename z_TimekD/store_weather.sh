#!/bin/bash
export OMP_NUM_THREADS=4
export PYTHONPATH=/home/intern/.local/lib/python3.8/site-packages:$PYTHONPATH

# data_paths=("ETTm1" "ETTm2")
data_paths=("weather")
divides=("train" "val") 
gpu_cnt=2
# num_nodes=7
num_nodes=21
input_len=96
output_len_values=(24 36 48 96 192)
model_name=("gpt2")
d_model=768
l_layer=12

accelerate config

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    for output_len in "${output_len_values[@]}"; do
      log_file="${data_path}_${output_len}_${divide}.log"
      accelerate launch --num_processes=$gpu_cnt store_emb.py \
        --data_path $data_path \
        --divide $divide \
        --num_nodes $num_nodes \
        --input_len $input_len \
        --output_len $output_len \
        --model_name $model_name \
        --d_model $d_model \
        --l_layer $l_layer > $log_file
    done
  done
done
