import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout, string_save, list_save
from visualization import plot_loss_curves

class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class DualErrorOutput:
    def __init__(self, filename):
        self.terminal = sys.stderr
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

def save_best_model_callback(
    monitor='val_loss',
    mode='min'
):
    best_value = None
    best_epoch = None
    def callback(model, loss, val_loss=None):
        nonlocal best_value, best_epoch
        current_value = val_loss if monitor == 'val_loss' else loss
        if best_value is None or (mode == 'min' and current_value < best_value) or (mode == 'max' and current_value > best_value):
            best_value = current_value
            best_epoch = model.n_epochs
            model.save(f'{run_dir}/model_best.pkl')
    return callback

    def get_best_epoch():
        return best_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    args = parser.parse_args()

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = DualOutput(f'{run_dir}/output.log')
    sys.stderr = DualErrorOutput(f'{run_dir}/error.log')
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    best_model_callback = save_best_model_callback(monitor='val_loss', mode='min')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        after_epoch_callback= best_model_callback
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
    
    t = time.time()
    
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    total_loss_log, loss_log, val_loss_log = model.fit(
        train_data,
        test_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')


    
    list_save(f'{run_dir}/total_loss.txt', total_loss_log)
    list_save(f'{run_dir}/loss.txt', loss_log)
    list_save(f'{run_dir}/val_loss.txt', val_loss_log)
    plot_loss_curves(f'{run_dir}/loss_curves', total_loss_log, loss_log, val_loss_log, cutoff=1)

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        model.load(f'{run_dir}/model_best.pkl')

        out_classification, eval_res_classification = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        os.makedirs(f'{run_dir}/classification', exist_ok=True)
        string_save(f'{run_dir}/classification/eval_res.txt', str(eval_res_classification))
        string_save(f'{run_dir}/classification/out.txt', str(out_classification))
        pkl_save(f'{run_dir}/classification/out.pkl', out_classification)
        pkl_save(f'{run_dir}/classification/eval_res.pkl', eval_res_classification)
        print("Classification evaluation results:")
        print(str(eval_res_classification))

        # out_forecasting, eval_res_forecasting = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        # os.makedirs(f'{run_dir}/forecasting', exist_ok=True)
        # string_save(f'{run_dir}/forecasting/eval_res.txt', str(eval_res_forecasting))
        # string_save(f'{run_dir}/forecasting/out.txt', str(out_forecasting))
        # pkl_save(f'{run_dir}/forecasting/out.pkl', out_forecasting)
        # pkl_save(f'{run_dir}/forecasting/eval_res.pkl', eval_res_forecasting)

        # out_anomaly_detection, eval_res_anomaly_detection = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        # os.makedirs(f'{run_dir}/anomaly_detection', exist_ok=True)
        # string_save(f'{run_dir}/anomaly_detection/eval_res.txt', str(eval_res_anomaly_detection))
        # string_save(f'{run_dir}/anomaly_detection/out.txt', str(out_anomaly_detection))
        # pkl_save(f'{run_dir}/anomaly_detection/out.pkl', out_anomaly_detection)
        # pkl_save(f'{run_dir}/anomaly_detection/eval_res.pkl', eval_res_anomaly_detection)

        # out_anomaly_detection_coldstart, eval_res_anomaly_detection_coldstart = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        # os.makedirs(f'{run_dir}/anomaly_detection_coldstart', exist_ok=True)
        # string_save(f'{run_dir}/anomaly_detection_coldstart/eval_res.txt', str(eval_res_anomaly_detection_coldstart))
        # string_save(f'{run_dir}/anomaly_detection_coldstart/out.txt', str(out_anomaly_detection_coldstart))
        # pkl_save(f'{run_dir}/anomaly_detection_coldstart/out.pkl', out_anomaly_detection_coldstart)
        # pkl_save(f'{run_dir}/anomaly_detection_coldstart/eval_res.pkl', eval_res_anomaly_detection_coldstart)

        os.makedirs(f'{run_dir}/clustering', exist_ok=True)
        out_clustering, eval_res_clustering = tasks.eval_clustering(f'{run_dir}/clustering', model, test_data, test_labels)
        
        string_save(f'{run_dir}/clustering/eval_res.txt', str(eval_res_clustering))
        string_save(f'{run_dir}/clustering/out.txt', str(out_clustering))
        pkl_save(f'{run_dir}/clustering/out.pkl', out_clustering)
        pkl_save(f'{run_dir}/clustering/eval_res.pkl', eval_res_clustering)
        print("Clustering evaluation results:")
        print(str(eval_res_clustering))

        

    print("Finished.")
