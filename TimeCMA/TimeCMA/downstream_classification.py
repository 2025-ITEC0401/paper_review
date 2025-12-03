import torch
import numpy as np
import argparse
import time
import os
import random
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from models.TimeCMA import Dual

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

class Dataset_Clustering(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1',
                 target='OT', scale=True, timeenc=0, freq='h',
                 model_name="gpt2"):
        if size == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[2] 

        # init
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
        
        self.model_name = model_name
        self.embed_path = f"/hdd/intern/keephun/TimeCMA/Embeddings/{data_path_file}/{flag}/"

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        try:
            df_raw = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"\n[Error] Cannot find file at: {self.data_path}")
            print(f"Please check --root_path argument.\n")
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

        if self.target in df_raw.columns:
            df_label = df_raw[self.target].astype(int) 
        else:
            df_label = df_raw.iloc[:, -1].astype(int)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_x = self.scaler.transform(df_data.values)
        else:
            data_x = df_data.values
            
        data_y = df_label.values 

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

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]
        label = self.data_y[s_end - 1] 
        seq_x_mark = self.data_stamp[s_begin:s_end]
        
        embeddings_stack = []
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if not os.path.exists(file_path):
            found_substitute = False
            for i in range(1, 5001): 
                prev_idx = index - i
                if prev_idx < 0: 
                    break 
                
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

        return seq_x, label, seq_x_mark, seq_x_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--data_path", type=str, required=True, help="dataset name")
    parser.add_argument("--root_path", type=str, default="./dataset/", help="root path of the data file")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="path to frozen model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./Results/classification/", help="output directory")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=192)
    parser.add_argument("--dropout_n", type=float, default=0.7)
    parser.add_argument("--d_llm", type=int, default=768)
    parser.add_argument("--e_layer", type=int, default=1)
    parser.add_argument("--d_layer", type=int, default=2)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="gpt2")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--feature_type", type=str, default="latent",
                        choices=["latent", "embedding", "raw"])
    parser.add_argument("--classifier", type=str, default="logistic",
                        choices=["logistic", "svm", "knn"])
    parser.add_argument("--target_col", type=str, default="OT")

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features(model, data_loader, device, feature_type="latent"):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch_data in data_loader:
            batch_x, batch_y, batch_x_mark, _, embeddings = batch_data
            
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
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
            labels_list.append(batch_y.view(-1).cpu().numpy())

    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    save_dir = os.path.join(args.output_dir, args.data_path, args.classifier)
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}...")
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

    print(f"Loading Datasets from {args.root_path} ...")
    
    train_dataset = Dataset_Clustering(
        root_path=args.root_path, 
        flag='train',
        size=[args.seq_len, 0, args.pred_len],
        features='M',
        data_path=args.data_path,
        target=args.target_col,
        model_name=args.model_name
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = Dataset_Clustering(
        root_path=args.root_path, 
        flag='test',
        size=[args.seq_len, 0, args.pred_len],
        features='M',
        data_path=args.data_path,
        target=args.target_col,
        model_name=args.model_name
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Extracting features ({args.feature_type})...")
    train_feats, train_labels = extract_features(model, train_loader, device, args.feature_type)
    test_feats, test_labels = extract_features(model, test_loader, device, args.feature_type)
    
    unique_classes = np.unique(train_labels)
    num_classes = len(unique_classes)

    print(f"Initializing Classifier ({args.classifier})...")
    if args.classifier == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=args.seed, n_jobs=-1)
    elif args.classifier == 'svm':
        clf = SVC(random_state=args.seed)
    elif args.classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    print("Calculating CV Accuracy...")
    cv_scores = cross_val_score(clf, train_feats, train_labels, cv=5, n_jobs=-1)
    cv_acc = np.mean(cv_scores)

    start = time.time()
    clf.fit(train_feats, train_labels)
    train_time = time.time() - start
    print(f"Training finished in {train_time:.2f} seconds.")

    preds = clf.predict(test_feats)
    
    acc = accuracy_score(test_labels, preds)
    f1_macro = f1_score(test_labels, preds, average='macro')
    f1_weighted = f1_score(test_labels, preds, average='weighted')

    print("\n" + "="*50)
    print(f"Classification Results: {args.data_path}")
    print("="*50)
    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 (Macro)    : {f1_macro:.4f}")
    print(f"F1 (Weighted) : {f1_weighted:.4f}")
    print(f"CV Accuracy   : {cv_acc:.4f}")
    print(f"Classes       : {num_classes} {unique_classes}")
    print("="*50)

    with open(os.path.join(save_dir, "metrics.txt"), 'w') as f:
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Feature: {args.feature_type}\n")
        f.write(f"Classifier: {args.classifier}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (Macro): {f1_macro:.4f}\n")
        f.write(f"F1 (Weighted): {f1_weighted:.4f}\n")
        f.write(f"CV Accuracy: {cv_acc:.4f}\n")
        f.write(f"Classes: {num_classes}\n")

    df_res = pd.DataFrame({
        'True_Label': test_labels,
        'Predicted_Label': preds,
        'Is_Correct': (test_labels == preds) 
    })
    df_res.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()