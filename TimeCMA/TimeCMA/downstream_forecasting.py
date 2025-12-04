import torch
import numpy as np
import argparse
import time
import os
import random
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from models.TimeCMA import Dual

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

class Dataset_Forecasting(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1',
                 target='OT', scale=True, timeenc=0, freq='h',
                 model_name="gpt2"):
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2] 

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        
        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv'
        else:
            data_path_file = data_path[:-4]

        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file
        
        # 임베딩 경로
        self.model_name = model_name
        self.embed_path = f"/hdd/intern/keephun/TimeCMA/Embeddings/{data_path_file}/{flag}/"

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        try:
            df_raw = pd.read_csv(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at: {self.data_path}")

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end] 
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        embeddings_stack = []
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        
        if not os.path.exists(file_path):
            found_substitute = False
            for i in range(1, 5001): 
                prev_idx = index - i
                if prev_idx < 0: break
                prev_path = os.path.join(self.embed_path, f"{prev_idx}.h5")
                if os.path.exists(prev_path):
                    file_path = prev_path
                    found_substitute = True
                    break
            if not found_substitute:
                file_path = os.path.join(self.embed_path, "0.h5")

        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                data = hf['embeddings'][:]
                tensor = torch.from_numpy(data)
                embeddings_stack.append(tensor.squeeze(0))
        else:
            dummy = torch.zeros((self.seq_len, 768))
            embeddings_stack.append(dummy)
                
        embeddings = torch.stack(embeddings_stack, dim=-1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./dataset/")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./Results/forecasting/")
    parser.add_argument('--seed', type=int, default=2024)
    
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction length")
    parser.add_argument("--dropout_n", type=float, default=0.7)
    parser.add_argument("--d_llm", type=int, default=768)
    parser.add_argument("--e_layer", type=int, default=1)
    parser.add_argument("--d_layer", type=int, default=2)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="gpt2")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--feature_type", type=str, default="latent", choices=["latent", "embedding", "raw"])
    parser.add_argument("--target_col", type=str, default="OT")

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def extract_features_and_targets(model, data_loader, device, feature_type="latent"):
    model.eval()
    features_list = []
    targets_list = []

    with torch.no_grad():
        for batch_data in data_loader:
            batch_x, batch_y, batch_x_mark, _, embeddings = batch_data
            
            batch_x = batch_x.float().to(device)
            embeddings = embeddings.float().to(device)

            if feature_type == "raw":
                normalized_data = model.normalize_layers(batch_x, 'norm')
                features = normalized_data.reshape(normalized_data.shape[0], -1) 
            elif feature_type == "embedding":
                embeddings_squeezed = embeddings.squeeze(-1) 
                features = embeddings_squeezed.reshape(embeddings_squeezed.shape[0], -1) 
            elif feature_type == "latent":
                input_data = model.normalize_layers(batch_x, 'norm')
                input_data = input_data.permute(0, 2, 1) 
                input_data = model.length_to_feature(input_data) 
                embeddings_squeezed = embeddings.squeeze(-1) 
                embeddings_squeezed = embeddings_squeezed.permute(0, 2, 1) 
                enc_out = model.ts_encoder(input_data) 
                enc_out = enc_out.permute(0, 2, 1) 
                prompt_enc = model.prompt_encoder(embeddings_squeezed) 
                prompt_enc = prompt_enc.permute(0, 2, 1) 
                cross_out = model.cross(enc_out, prompt_enc, prompt_enc) 
                features = cross_out.reshape(cross_out.shape[0], -1) 

            features_list.append(features.cpu().numpy())
            targets_list.append(batch_y.numpy())

    return np.concatenate(features_list, axis=0), np.concatenate(targets_list, axis=0)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    save_dir = os.path.join(args.output_dir, args.data_path)
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading TimeCMA model...")
    model = Dual(
        device=device, channel=args.channel, num_nodes=args.num_nodes, seq_len=args.seq_len,
        pred_len=args.pred_len, dropout_n=args.dropout_n, d_llm=args.d_llm, e_layer=args.e_layer,
        d_layer=args.d_layer, head=args.head
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loading Datasets from {args.root_path}...")
    train_dataset = Dataset_Forecasting(
        root_path=args.root_path, flag='train', size=[args.seq_len, 0, args.pred_len],
        features='M', data_path=args.data_path, target=args.target_col, model_name=args.model_name
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = Dataset_Forecasting(
        root_path=args.root_path, flag='test', size=[args.seq_len, 0, args.pred_len],
        features='M', data_path=args.data_path, target=args.target_col, model_name=args.model_name
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Extracting features ({args.feature_type})...")
    train_feats, train_targets = extract_features_and_targets(model, train_loader, device, args.feature_type)
    test_feats, test_targets = extract_features_and_targets(model, test_loader, device, args.feature_type)
    
    N_train, P_len, N_nodes = train_targets.shape
    train_targets_flat = train_targets.reshape(N_train, -1)
    
    N_test, _, _ = test_targets.shape
    test_targets_flat = test_targets.reshape(N_test, -1)

    # 4. Train Regressor (Ridge Regression)
    print("Training Regressor (Ridge)...")
    regressor = Ridge(alpha=1.0)
    start = time.time()
    regressor.fit(train_feats, train_targets_flat)
    print(f"Training finished in {time.time() - start:.2f} seconds.")

    print("Predicting...")
    preds_flat = regressor.predict(test_feats)
    
    preds = preds_flat.reshape(N_test, P_len, N_nodes)
    trues = test_targets # Original shape

    print("Inverse Transforming...")
    preds_inv = test_dataset.scaler.inverse_transform(preds.reshape(-1, N_nodes)).reshape(N_test, P_len, N_nodes)
    trues_inv = test_dataset.scaler.inverse_transform(trues.reshape(-1, N_nodes)).reshape(N_test, P_len, N_nodes)

    mse = mean_squared_error(trues_inv.reshape(-1), preds_inv.reshape(-1))
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues_inv.reshape(-1), preds_inv.reshape(-1))
    r2 = r2_score(trues_inv.reshape(-1), preds_inv.reshape(-1))

    print("\n" + "="*50)
    print(f"Forecasting Results: {args.data_path}")
    print("="*50)
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print("="*50)

    with open(os.path.join(save_dir, "forecast_metrics.txt"), 'w') as f:
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Feature: {args.feature_type}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")

    plt.figure(figsize=(10, 5))
    plt.plot(trues_inv[0, :, 0], label='GroundTruth')
    plt.plot(preds_inv[0, :, 0], label='Prediction')
    plt.title(f"Forecast Sample (Node 0) - {args.data_path}")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "forecast_sample.png"))
    plt.close()

    print(f"Results and visual saved to {save_dir}")

if __name__ == "__main__":
    main()