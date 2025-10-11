#!/bin/bash
export PYTHONPATH=/home/intern/.local/lib/python3.8/site-packages:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

device="cuda:1"
learning_rates=(1e-4 1e-5)
batch_sizes=(16)
seq_lens=(96)     # Input Length
pred_lens=(24 36 48 96 192) # Output Length
channels=(64)
d_llm=(768)
e_layers=(2)
dropout_ns=(0.5)
model_name="gpt2"
data_paths=("exchange_rate")
epochs=(100)
num_node=(8)

for data_path in "${data_paths[@]}"; do
  for seq_len in "${seq_lens[@]}"; do 
    for pred_len in "${pred_lens[@]}"; do
      for learning_rate in "${learning_rates[@]}"; do
        for channel in "${channels[@]}"; do
          for dropout_n in "${dropout_ns[@]}"; do
            for e_layer in "${e_layers[@]}"; do
              for batch_size in "${batch_sizes[@]}"; do
                log_path="./Results/Fcst/${data_path}/"
                mkdir -p $log_path
                log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dn${dropout_n}_bs${batch_size}_e${epochs}.log"
                python3 train.py \
                  --data_path $data_path \
                  --device $device \
                  --batch_size $batch_size \
                  --num_nodes ${num_node} \
                  --seq_len $seq_len \
                  --pred_len $pred_len \
                  --epochs $epochs \
                  --seed 6666 \
                  --channel $channel \
                  --head 4 \
                  --lrate $learning_rate \
                  --dropout_n $dropout_n \
                  --e_layer $e_layer\
                  --model_name $model_name \
                  --num_workers 10 \
                  --d_llm $d_llm > $log_file
              done
            done
          done
        done
      done
    done
  done
done
