import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.datasets._data_io import load_from_tsfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import warnings

from models.PatchTST import Model as PatchTST

class PatchTSTClassifier(nn.Module):
    def __init__(self, args, num_classes):
        super(PatchTSTClassifier, self).__init__()
        args.task_name = 'reconstruction'
        self.backbone = PatchTST(args)
        self.flatten_dim = args.seq_len * args.enc_in
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x_feat = self.backbone(x, None, None, None)
        logits = self.head(x_feat)
        return logits

def load_data(dataset_name, root_path):
    train_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    print(f"--- Loading {dataset_name}... ---")
    X_train_df, y_train = load_from_tsfile(train_file)
    X_test_df, y_test = load_from_tsfile(test_file)

    def to_numpy_3d_padded(df):
        n_samples = df.shape[0]
        n_channels = df.shape[1]
        max_len = 0
        for i in range(n_samples):
            for j in range(n_channels):
                max_len = max(max_len, df.iloc[i, j].shape[0])
        arr = np.zeros((n_samples, max_len, n_channels), dtype=np.float32)
        for i in range(n_samples):
            for j in range(n_channels):
                series = df.iloc[i, j].to_numpy()
                arr[i, :series.shape[0], j] = series
        return arr

    X_train = to_numpy_3d_padded(X_train_df)
    X_test = to_numpy_3d_padded(X_test_df)
    
    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train)
    y_test_int = le.transform(y_test) # Handle unseen labels if necessary
    
    # Validation for unseen labels in test set
    # (Simple fix: map unseen to 0 or handle error. Here we assume standard UCR splits)
    
    scaler = StandardScaler()
    n_train, t_len, n_ch = X_train.shape
    n_test, _, _ = X_test.shape
    
    X_train_flat = X_train.reshape(-1, n_ch)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, t_len, n_ch)
    X_test_flat = X_test.reshape(-1, n_ch)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_test, t_len, n_ch)

    return X_train_scaled, y_train_int, X_test_scaled, y_test_int, len(le.classes_), le.classes_

def run_classification(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device: {device} ---")
    torch.cuda.empty_cache()

    X_train, y_train, X_test, y_test, num_classes, class_names = load_data(args.dataset_name, args.root_path)
    
    real_seq_len = X_train.shape[1]
    real_enc_in = X_train.shape[2]
    
    print(f"[Auto-Config] Seq Len: {real_seq_len}, Enc In: {real_enc_in}, Classes: {num_classes}")
    
    # -----------------------------------------------------------
    # [Fix] PenDigits 처럼 길이가 짧은 데이터셋을 위한 패치 크기 자동 조절
    # -----------------------------------------------------------
    if real_seq_len < args.patch_len:
        new_patch_len = max(1, real_seq_len // 2) if real_seq_len > 1 else 1
        new_stride = max(1, new_patch_len // 2)
        print(f"[Auto-Fix] SeqLen({real_seq_len}) < PatchLen({args.patch_len}). Adjusting -> Patch: {new_patch_len}, Stride: {new_stride}")
        args.patch_len = new_patch_len
        args.stride = new_stride
    
    args.seq_len = real_seq_len
    args.pred_len = real_seq_len
    args.enc_in = real_enc_in
    args.context_window = real_seq_len
    args.c_out = real_enc_in

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = PatchTSTClassifier(args, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[Training Start] {args.dataset_name}")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        train_acc = []
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch_y).float().mean().item()
            train_acc.append(acc)
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {np.mean(train_loss):.4f} | Acc: {np.mean(train_acc):.4f}")

    print(f"\n[Evaluation Start]")
    model.eval()
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(batch_y.cpu().numpy())
            torch.cuda.empty_cache()

    acc = accuracy_score(all_trues, all_preds)
    f1_macro = f1_score(all_trues, all_preds, average='macro')
    f1_weighted = f1_score(all_trues, all_preds, average='weighted')
    cm = confusion_matrix(all_trues, all_preds)
    # 0으로 나누기 방지
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)
    
    print("\n" + "="*50)
    print(f" Dataset: {args.dataset_name}")
    print("-" * 50)
    print(f" 1. Accuracy       : {acc:.4f}")
    print(f" 2. F1 (Macro)     : {f1_macro:.4f}")
    print(f" 3. F1 (Weighted)  : {f1_weighted:.4f}")
    print(f" 4. CV (Test Acc)  : {acc:.4f}")
    print("-" * 50)
    print(" 5. Accuracy per Class:")
    for i, class_acc in enumerate(per_class_acc):
        c_name = str(i)
        if i < len(class_names): c_name = str(class_names[i])
        print(f"    - Class '{c_name}': {class_acc:.4f}")
    print("="*50 + "\n")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {args.dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"CM_{args.dataset_name}.png")
    plt.close()
    print(f"   -> [Confusion Matrix Saved]: CM_{args.dataset_name}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--fc_dropout', type=float, default=0.0)
    parser.add_argument('--padding_patch', default='end')
    parser.add_argument('--revin', type=int, default=0)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--individual', type=int, default=1)
    parser.add_argument('--embed_type', type=int, default=0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=1)
    args = parser.parse_args()
    run_classification(args)
