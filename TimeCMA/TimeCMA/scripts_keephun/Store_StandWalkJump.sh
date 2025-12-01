 #!/bin/bash
export PYTHONPATH=/hdd/intern/daniel/TimeCMA/sfs-common/cxliu/TimeCMA:$PYTHONPATH


data_paths=("StandWalkJump_m")
divides=("train" "val" "test")
num_nodes=3
input_len=96
output_len=96

for data_path in "${data_paths[@]}"; do
  divide="train"
  log_file="./Results/emb_logs/${data_path}_${divide}.log"
  export CUDA_VISIBLE_DEVICES=1
  nohup /hdd/conda_envs/envs/TimeCMA/bin/python storage/store_emb.py --divide $divide --data_path $data_path --num_nodes $num_nodes --input_len $input_len --output_len $output_len > $log_file &

  divide="val"
  log_file="./Results/emb_logs/${data_path}_${divide}.log"
  export CUDA_VISIBLE_DEVICES=1
  nohup /hdd/conda_envs/envs/TimeCMA/bin/python storage/store_emb.py --divide $divide --data_path $data_path --num_nodes $num_nodes --input_len $input_len --output_len $output_len > $log_file &

  divide="test"
  log_file="./Results/emb_logs/${data_path}_${divide}.log"
  export CUDA_VISIBLE_DEVICES=1
  nohup /hdd/conda_envs/envs/TimeCMA/bin/python storage/store_emb.py --divide $divide --data_path $data_path --num_nodes $num_nodes --input_len $input_len --output_len $output_len > $log_file &
done