
batch_size=8
repr_dims=320
max_threads=8
seed=42
epochs=1000

nohup /hdd/conda_envs/envs/ts2vec_daniel/bin/python -u train.py BasicMotions UEA --loader UEA --batch-size $batch_size --repr-dims $repr_dims --max-threads $max_threads --seed $seed --epochs $epochs --gpu 0 --eval >/dev/null &
nohup /hdd/conda_envs/envs/ts2vec_daniel/bin/python -u train.py Epilepsy UEA --loader UEA --batch-size $batch_size --repr-dims $repr_dims --max-threads $max_threads --seed $seed --epochs $epochs --gpu 1 --eval >/dev/null &
nohup /hdd/conda_envs/envs/ts2vec_daniel/bin/python -u train.py HandMovementDirection UEA --loader UEA --batch-size $batch_size --repr-dims $repr_dims --max-threads $max_threads --seed $seed --gpu 0 --epochs $epochs --eval >/dev/null &
nohup /hdd/conda_envs/envs/ts2vec_daniel/bin/python -u train.py Libras UEA --loader UEA --batch-size $batch_size --repr-dims $repr_dims --max-threads $max_threads --seed $seed --epochs $epochs --gpu 1 --eval >/dev/null &