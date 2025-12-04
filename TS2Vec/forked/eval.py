import torch
import numpy as np
import argparse
import os
import datetime
import sys
import time
import shutil
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, pkl_save, string_save

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model (.pkl file)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save evaluation results (defaults to same directory as model)')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for inference (defaults to 0)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval-protocol', type=str, default='svm', help='Evaluation protocol for classification: linear, svm, knn (defaults to svm)')
    parser.add_argument('--tasks', type=str, default='classification,clustering', help='Comma-separated list of tasks to evaluate: classification, clustering, imputation (defaults to classification,clustering)')
    parser.add_argument('--missing-ratios', type=str, default='0.1,0.2,0.3,0.4,0.5', help='Comma-separated missing ratios for imputation task (defaults to 0.1,0.2,0.3,0.4,0.5)')
    parser.add_argument('--missing-types', type=str, default='random,block', help='Comma-separated missing types for imputation task: random, block, feature (defaults to random,block)')
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.model_path)
        if not output_dir:
            output_dir = '.'
        output_dir = os.path.join(output_dir, 'eval_results')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(f'{output_dir}/eval_output.log'):
        os.remove(f'{output_dir}/eval_output.log')
    if os.path.exists(f'{output_dir}/eval_error.log'):
        os.remove(f'{output_dir}/eval_error.log')
    sys.stdout = DualOutput(f'{output_dir}/eval_output.log')
    sys.stderr = DualErrorOutput(f'{output_dir}/eval_error.log')
    
    print("=" * 60)
    print("Evaluation Script")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Arguments: {str(args)}")
    print("=" * 60)
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'UEA_forecast':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_UEA_forecast(args.dataset)
        train_data = data[:, train_slice]
        
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
    
    elif args.loader == 'UEA_anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_UEA_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
    print('done')
    
    # Load model
    print('Loading model... ', end='')
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        output_dims=args.repr_dims
    )
    model.load(args.model_path)
    print('done')
    
    # Parse tasks
    eval_tasks = [t.strip() for t in args.tasks.split(',')]
    
    t = time.time()
    
    # Classification evaluation
    if 'classification' in eval_tasks:
        if task_type == 'classification' or args.loader == 'UEA_forecast':
            print("\n" + "=" * 40)
            print("Running Classification Evaluation...")
            print("=" * 40)
            out_classification, eval_res_classification = tasks.eval_classification(
                model, train_data, train_labels, test_data, test_labels, 
                eval_protocol=args.eval_protocol
            )
        
        if os.path.exists(f'{output_dir}/classification'):
            shutil.rmtree(f'{output_dir}/classification')
        os.makedirs(f'{output_dir}/classification', exist_ok=True)
        string_save(f'{output_dir}/classification/eval_res.txt', str(eval_res_classification))
        string_save(f'{output_dir}/classification/out.txt', str(out_classification))
        pkl_save(f'{output_dir}/classification/out.pkl', out_classification)
        pkl_save(f'{output_dir}/classification/eval_res.pkl', eval_res_classification)
        
        print("Classification evaluation results:")
        for key, value in eval_res_classification.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Clustering evaluation
    if 'clustering' in eval_tasks:
        print("\n" + "=" * 40)
        print("Running Clustering Evaluation...")
        print("=" * 40)
        
        if os.path.exists(f'{output_dir}/clustering'):
            shutil.rmtree(f'{output_dir}/clustering')

        os.makedirs(f'{output_dir}/clustering', exist_ok=True)
        out_clustering, eval_res_clustering = tasks.eval_clustering(
            f'{output_dir}/clustering', model, test_data, test_labels
        )
        
        string_save(f'{output_dir}/clustering/eval_res.txt', str(eval_res_clustering))
        string_save(f'{output_dir}/clustering/out.txt', str(out_clustering))
        pkl_save(f'{output_dir}/clustering/out.pkl', out_clustering)
        pkl_save(f'{output_dir}/clustering/eval_res.pkl', eval_res_clustering)
        
        print("Clustering evaluation results:")
        print(str(eval_res_clustering))
    
    # Forecasting evaluation
    if 'forecasting' in eval_tasks:
        if task_type == 'forecasting' or args.loader == 'UEA_forecast':
            print("\n" + "=" * 40)
            print("Running Forecasting Evaluation...")
            print("=" * 40)
            
            out_forecasting, eval_res_forecasting = tasks.eval_forecasting(
                model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols
            )
            
            if os.path.exists(f'{output_dir}/forecasting'):
                shutil.rmtree(f'{output_dir}/forecasting')
            os.makedirs(f'{output_dir}/forecasting', exist_ok=True)
            string_save(f'{output_dir}/forecasting/eval_res.txt', str(eval_res_forecasting))
            string_save(f'{output_dir}/forecasting/out.txt', str(out_forecasting))
            pkl_save(f'{output_dir}/forecasting/out.pkl', out_forecasting)
            pkl_save(f'{output_dir}/forecasting/eval_res.pkl', eval_res_forecasting)
            
            print("Forecasting evaluation results:")
            print(str(eval_res_forecasting))
        else:
            print("\n" + "=" * 40)
            print("Skipping Forecasting Evaluation (not supported for this data loader)")
            print("=" * 40)
    
    # Anomaly detection evaluation
    if 'anomaly_detection' in eval_tasks:
        if task_type in ('anomaly_detection', 'anomaly_detection_coldstart'):
            print("\n" + "=" * 40)
            print("Running Anomaly Detection Evaluation...")
            print("=" * 40)
            
            out_anomaly_detection, eval_res_anomaly_detection = tasks.eval_anomaly_detection(
                model, all_train_data, all_train_labels, all_train_timestamps, 
                all_test_data, all_test_labels, all_test_timestamps, delay
            )
            
            if os.path.exists(f'{output_dir}/anomaly_detection'):
                shutil.rmtree(f'{output_dir}/anomaly_detection')
            os.makedirs(f'{output_dir}/anomaly_detection', exist_ok=True)
            string_save(f'{output_dir}/anomaly_detection/eval_res.txt', str(eval_res_anomaly_detection))
            string_save(f'{output_dir}/anomaly_detection/out.txt', str(out_anomaly_detection))
            pkl_save(f'{output_dir}/anomaly_detection/out.pkl', out_anomaly_detection)
            pkl_save(f'{output_dir}/anomaly_detection/eval_res.pkl', eval_res_anomaly_detection)
            
            print("Anomaly detection evaluation results:")
            print(str(eval_res_anomaly_detection))
        else:
            print("\n" + "=" * 40)
            print("Skipping Anomaly Detection Evaluation (not supported for this data loader)")
            print("=" * 40)
    
    # Imputation evaluation
    if 'imputation' in eval_tasks:
        print("\n" + "=" * 40)
        print("Running Imputation Evaluation...")
        print("=" * 40)
        
        # Parse missing ratios and types
        missing_ratios = [float(r.strip()) for r in args.missing_ratios.split(',')]
        missing_types = [t.strip() for t in args.missing_types.split(',')]
        
        # Use test_data for imputation evaluation if available
        if 'test_data' in dir():
            imputation_data = test_data
        else:
            imputation_data = train_data
        
        imputation_results, imputation_summary = tasks.eval_imputation(
            model, imputation_data, 
            missing_ratios=missing_ratios,
            missing_types=missing_types,
            device=device
        )

        if os.path.exists(f'{output_dir}/imputation'):
            shutil.rmtree(f'{output_dir}/imputation')
        
        os.makedirs(f'{output_dir}/imputation', exist_ok=True)
        string_save(f'{output_dir}/imputation/eval_res.txt', str(imputation_results))
        string_save(f'{output_dir}/imputation/summary.txt', str(imputation_summary))
        pkl_save(f'{output_dir}/imputation/eval_res.pkl', imputation_results)
        pkl_save(f'{output_dir}/imputation/summary.pkl', imputation_summary)
        
        tasks.print_imputation_results(imputation_results, imputation_summary)
    
    t = time.time() - t
    print(f"\nEvaluation time: {datetime.timedelta(seconds=t)}")
    print("\nFinished.")
