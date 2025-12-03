import argparse
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from sktime.datasets._data_io import load_from_tsfile
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 기존 모델 파일 불러오기
from models.PatchTST import Model as PatchTST

# -----------------------------------------------------------------------------
# 1. 데이터 로딩 및 전처리 (바이너리 라벨링 포함)
# -----------------------------------------------------------------------------
def load_data(dataset_name, root_path):
    train_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(root_path, dataset_name, f"{dataset_name}_TEST.ts")
    
    if not os.path.exists(train_file):
        print(f"[Error] {train_file} not found.")
        return None, None, None, None

    # sktime으로 로드
    X_train_df, y_train = load_from_tsfile(train_file)
    X_test_df, y_test = load_from_tsfile(test_file)

    # 3D Numpy 변환 함수
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

    # ----------------------------------------------------------
    # [중요] 이상 탐지를 위한 라벨 변환 (Normal=0, Anomaly=1)
    # ----------------------------------------------------------
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    y_train_int = le.transform(y_train)
    y_test_int = le.transform(y_test)

    # Train 데이터에서 가장 많은 클래스를 '정상'으로 가정
    vals, counts = np.unique(y_train_int, return_counts=True)
    normal_class = vals[np.argmax(counts)]
    print(f"   > '{dataset_name}' Normal Class Assumption: {le.inverse_transform([normal_class])[0]} (Index: {normal_class})")

    # Binary Label 생성 (Normal=0, Anomaly=1)
    y_train_bin = np.where(y_train_int == normal_class, 0, 1)
    y_test_bin = np.where(y_test_int == normal_class, 0, 1)

    return X_train, y_train_bin, X_test, y_test_bin

# -----------------------------------------------------------------------------
# 2. PatchTST를 이용한 Feature Extraction
# -----------------------------------------------------------------------------
def extract_features(model, data, device, batch_size=16):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features_list = []
    
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            
            # 모델 실행 (features 추출을 위해 try-except 사용)
            try:
                # PatchTST가 Classification 모드일 때 output이 (Batch, NumClass)일 수 있음
                # 하지만 여기선 Feature가 필요하므로, 가능하다면 model.encoder() 등을 써야 함.
                # 현재는 model(x) 결과를 씁니다.
                outputs = model(batch_x)
                
                # 출력이 3차원(B, Time, Channel)이라면 평균 -> (B, Channel)
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1) 
                # 출력이 4차원 이상이면 Flatten
                elif outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                    
            except Exception as e:
                # 에러 시 원본 데이터 Flatten (Fallback)
                print(f"Warning: Model forward failed ({e}), using raw data.")
                outputs = batch_x.view(batch_x.size(0), -1)

            features_list.append(outputs.cpu().numpy())
            
    return np.concatenate(features_list, axis=0)

# -----------------------------------------------------------------------------
# 3. Anomaly Detection 실행 및 평가
# -----------------------------------------------------------------------------
def run_anomaly_detection(train_feats, test_feats, y_test, method='isolation_forest'):
    print(f"   > Running {method}...")
    
    if method == 'isolation_forest':
        clf = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    elif method == 'one_class_svm':
        clf = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
    else:
        raise ValueError("Unknown method")

    # 학습
    clf.fit(train_feats)

    # 예측 및 점수 계산
    # Decision Function: 양수=정상, 음수=이상이 일반적 -> 부호 반전하여 '이상 점수'로 활용
    y_scores = clf.decision_function(test_feats) 
    
    # Prediction: 1=정상, -1=이상 -> 변환: 0=정상, 1=이상
    y_pred_raw = clf.predict(test_feats)
    y_pred = np.where(y_pred_raw == 1, 0, 1)
    
    # ROC-AUC용 Score (높을수록 이상치일 확률이 높도록)
    if method == 'isolation_forest':
        score = -clf.decision_function(test_feats)
    else:
        score = -y_scores

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, score)
    except:
        auc = 0.5 

    return acc, prec, rec, f1, auc

# -----------------------------------------------------------------------------
# 메인 실행 함수
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--gpu', type=int, default=0)
    
    # [수정된 부분] task_name 추가 (모델 초기화에 필수)
    parser.add_argument('--task_name', type=str, default='classification')

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

    # 실험할 데이터셋 목록 (DatasetName, SeqLen, EncIn)
    datasets_config = [
        ("AtrialFibrillation", 640, 2),
        ("PEMS-SF", 963, 963),
        ("StandWalkJump", 2500, 4)
    ]

    print("=== PatchTST Feature based Anomaly Detection (IF & OCSVM) ===")

    for name, seq_len, enc_in in datasets_config:
        print(f"\n[Dataset: {name}] Loading...")
        
        X_train, y_train, X_test, y_test = load_data(name, args.root_path)
        if X_train is None: continue

        # args 업데이트
        args.seq_len = seq_len
        args.enc_in = enc_in
        args.c_out = enc_in 
        
        # 모델 초기화를 위해 임시 클래스 수 설정 (실제 분류는 안 하므로 무관)
        args.num_class = 2 
        
        try:
            model = PatchTST(args).float().to(device)
        except Exception as e:
            print(f"Error initializing model: {e}")
            continue
        
        print("   > Extracting Features using PatchTST...")
        train_feats = extract_features(model, X_train, device)
        test_feats = extract_features(model, X_test, device)
        
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        test_feats = scaler.transform(test_feats)

        # 4. Isolation Forest
        acc, prec, rec, f1, auc = run_anomaly_detection(train_feats, test_feats, y_test, method='isolation_forest')
        print(f"   [Isolation Forest] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # 5. One-Class SVM
        if name == 'PEMS-SF' and train_feats.shape[0] > 10000:
            print("   [One-Class SVM] Skipped for PEMS-SF (Too large)")
        else:
            acc, prec, rec, f1, auc = run_anomaly_detection(train_feats, test_feats, y_test, method='one_class_svm')
            print(f"   [One-Class SVM ] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    print("\n=== All Experiments Completed ===")