 #!/bin/bash

/hdd/conda_envs/envs/TimeCMA/bin/python storage/store_emb.py --divide "train" --data_path "ETTh1" --num_nodes 7 --input_len 96 --output_len 96 > ./Results/emb_logs/ETTh1_train.log &