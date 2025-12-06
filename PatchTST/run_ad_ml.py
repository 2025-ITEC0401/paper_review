import argparse
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sktime.datasets._data_io import load_from_tsfile
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# 기존 모델 파일 불러오기
from models.PatchTST import Model as PatchTST

# -----------------------------------------------------------------------------
# 0. 시각화 함수
# -----------------------------------------------------------------------------
def visualize_ad_results(y_true, y_scores, y_pred, dataset_name, method_name):
    plt.figure(figsize=(18, 5))
    
    # 1. ROC Curve
    plt.subplot(1, 3, 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    try:
        auc_score = roc_auc_score(y_true, y_scores)
    except:
        auc_score = 0.5
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({dataset_name} - {method_name})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # 2. Anomaly Score Histogram
    plt.subplot(1, 3, 2)
    sns.histplot(x=y_scores, hue=y_true, bins=50, kde=True, element="step", stat="density", common_norm=False, palette={0: 'blue', 1: 'red'})
    plt.title(f'Anomaly Score Distribution\n(Blue: Normal, Red: Anomaly)')
    plt.xlabel('Anomaly Score (Higher = More Anomalous)')
    
    # 3. Confusion Matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Pred Normal', 'Pred Anomaly'], 
                yticklabels=['True Normal', 'True Anomaly'])
    plt.title(f'Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.tight_layout()
    save_name = f"AD_result_{dataset_name}_{method_name}.png"
    plt.savefig(save_name)
    plt.close()
    print(f"   -> 시각화 저장 완료: {save_name}")

# -----------------------------------------------------------------------------
# 1. 데이터 로딩 및 전처리
# -----------------------------------------------------------------------------
def load_data(dataset_name, root_path):
    train_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    
    if not os.path.exists(train_file):
        print(f"[Error] {train_file} not found.")
        return None, None, None, None

    # 경고 무시
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    X_train_df, y_train = load_from_tsfile(train_file)
    X_test_df, y_test = load_from_tsfile(test_file)

    def to_numpy_3d(df):
        n_samples = df.shape[0]
        n_channels = df.shape[1]
        n_timesteps = df.iloc[0, 0].shape[0]
        arr = np.empty((n_samples, n_timesteps, n_channels), dtype=np.float32)
        for i in range(n_samples):
            for j in range(n_channels):
                arr[i, :, j] = df.iloc[i, j].to_numpy()
        return arr

    X_train = to_numpy_3d(X_train_df)
    X_test = to_numpy_3d(X_test_df)

    # 라벨링
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    y_train_int = le.transform(y_train)
    y_test_int = le.transform(y_test)

    vals, counts = np.unique(y_train_int, return_counts=True)
    normal_class = vals[np.argmax(counts)]
    print(f"   > '{dataset_name}' Normal Class Assumption: {le.inverse_transform([normal_class])[0]} (Index: {normal_class})")

    y_train_bin = np.where(y_train_int == normal_class, 0, 1)
    y_test_bin = np.where(y_test_int == normal_class, 0, 1)

    return X_train, y_train_bin, X_test, y_test_bin

# -----------------------------------------------------------------------------
# 2. Feature Extraction
# -----------------------------------------------------------------------------
def extract_features(model, data, device, batch_size=16):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features_list = []
    
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            try:
                outputs = model(batch_x)
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1) 
                elif outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
            except:
                outputs = batch_x.view(batch_x.size(0), -1)
            features_list.append(outputs.cpu().numpy())
            
    return np.concatenate(features_list, axis=0)

# -----------------------------------------------------------------------------
# 3. Anomaly Detection 실행
# -----------------------------------------------------------------------------
def run_anomaly_detection(train_feats, test_feats, y_test, dataset_name, method='isolation_forest'):
    print(f"   > Running {method}...")
    
    if method == 'isolation_forest':
        clf = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    elif method == 'one_class_svm':
        clf = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
    else:
        raise ValueError("Unknown method")

    clf.fit(train_feats)
    
    y_scores_raw = clf.decision_function(test_feats)
    y_scores = -y_scores_raw 

    y_pred_raw = clf.predict(test_feats)
    y_pred = np.where(y_pred_raw == 1, 0, 1)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_scores)
    except:
        auc = 0.5 

    visualize_ad_results(y_test, y_scores, y_pred, dataset_name, method)

    return acc, prec, rec, f1, auc

# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_name', type=str, default='classification')
    parser.add_argument('--pred_len', type=int, default=24) 

    # PatchTST Parameters
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=1)
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
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    datasets_config = [
        ("AtrialFibrillation", 640, 2),
        ("PEMS-SF", 963, 963),
        ("StandWalkJump", 2500, 4)
    ]

    print("=== PatchTST Anomaly Detection with Visualization ===")

    for name, seq_len, enc_in in datasets_config:
        print(f"\n[Dataset: {name}] Processing...")
        
        X_train, y_train, X_test, y_test = load_data(name, args.root_path)
        if X_train is None: continue

        args.seq_len = seq_len
        args.enc_in = enc_in
        args.c_out = enc_in 
        args.num_class = 2 
        
        try:
            model = PatchTST(args).float().to(device)
        except Exception as e:
            print(f"Error initializing model: {e}")
            continue
        
        print("   > Extracting Features...")
        batch_s = 4 if name == 'PEMS-SF' else 16
        train_feats = extract_features(model, X_train, device, batch_size=batch_s)
        test_feats = extract_features(model, X_test, device, batch_size=batch_s)
        
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        test_feats = scaler.transform(test_feats)

        # 4. Isolation Forest
        acc, prec, rec, f1, auc = run_anomaly_detection(train_feats, test_feats, y_test, name, method='isolation_forest')
        # [수정됨] 모든 지표 출력
        print(f"   [Isolation Forest] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # 5. One-Class SVM
        if name == 'PEMS-SF' and train_feats.shape[0] > 5000:
            print("   [One-Class SVM] Skipped (Too large)")
        else:
            acc, prec, rec, f1, auc = run_anomaly_detection(train_feats, test_feats, y_test, name, method='one_class_svm')
            # [수정됨] 모든 지표 출력
            print(f"   [One-Class SVM ] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    print("\n=== All Experiments & Visualizations Completed ===")