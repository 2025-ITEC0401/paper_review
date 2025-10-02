#!/bin/bash
export PYTHONPATH=/home/intern/.local/lib/python3.8/site-packages:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

device="cuda:0"
seq_lens=(96)
pred_lens=(24)
learning_rates=(1e-4 1e-5)
channels=(64)
d_llm=(768)
e_layers=(2)
dropout_ns=(0.5)
batch_sizes=(16)
model_name="gpt2"
data_paths=("ETTm1" "ETTm2" "ETTh1" "ETTh2")
epochs=(100)

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
                nohup python3 train.py \
                  --data_path $data_path \
                  --device $device \
                  --batch_size $batch_size \
                  --num_nodes 7 \
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
                  --d_llm $d_llm > $log_file &
              done
            done
          done
        done
      done
    done
  done
done
