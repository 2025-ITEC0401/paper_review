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

# 기존 모델 파일 불러오기 (경로에 맞게 수정 필요)
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
    # 데이터셋마다 '정상' 클래스의 정의가 다를 수 있습니다.
    # 여기서는 가장 빈도가 높은 클래스를 'Normal(0)'로 가정합니다.
    # 실제 도메인 지식이 있다면 이 부분을 수정하세요.
    # ----------------------------------------------------------
    
    # 1. 모든 라벨을 정수로 인코딩
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    y_train_int = le.transform(y_train)
    y_test_int = le.transform(y_test)

    # 2. '정상' 클래스 식별 (Train 데이터에서 가장 많은 클래스)
    vals, counts = np.unique(y_train_int, return_counts=True)
    normal_class = vals[np.argmax(counts)]
    print(f"   > '{dataset_name}' Normal Class Assumption: {le.inverse_transform([normal_class])[0]} (Index: {normal_class})")

    # 3. Binary Label 생성 (Normal=0, Anomaly=1)
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
            
            # PatchTST Encoder의 출력(Representation)을 가져옵니다.
            # 모델 구조에 따라 model.forward()가 아니라 model.encoder()를 호출해야 할 수도 있습니다.
            # 여기서는 일반적인 PatchTST 구현체의 encoder 출력을 가정합니다.
            # (Batch, N_patches, D_model) 형태라고 가정
            
            # 1. PatchTST 모델 내부에 encoder 접근이 가능한지 확인 필요
            # 보통 output = model(batch_x)는 prediction 결과이므로,
            # 내부 코드를 수정하거나 아래처럼 hook/별도 메소드를 써야 함.
            # 편의상 model(batch_x)가 classification head 직전의 embedding을 줄 수 있다고 가정하거나
            # 혹은 model.encoder(...)를 호출.
            
            try:
                # PatchTST 구조에 따라 다름. 여기서는 임시로 모델 forward 결과를 사용하거나
                # 인코더가 있다면 인코더를 통과시킴. 
                # 만약 model이 forecasting용이면 output은 (B, L, C) 예측값임.
                # 이 경우 예측값과 실제값의 차이(Error)를 feature로 쓸 수도 있으나,
                # 요청하신 건 "특징 기반" 머신러닝이므로 Representation이 필요함.
                
                # [가정] model 객체에 enc_embedding 같은 속성이 있거나 
                # forward 시 output_hidden_states=True 옵션이 있다고 가정.
                # 여기서는 간단히 '모델의 출력'을 Flatten해서 사용 (구조에 따라 수정 필요)
                
                # *수정 제안*: forecasting 모델이라면, model.forward 대신 
                # model.backbone(...) 혹은 model.encoder(...)를 호출해야 함.
                # 일단 예시로 model(x) 결과를 씁니다.
                
                outputs = model(batch_x) 
                
                # 출력이 3차원(B, Time, Channel)이라면 평균내서 2차원으로 만듦
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1) 
                elif outputs.dim() > 3:
                    outputs = outputs.view(outputs.size(0), -1)
                    
            except:
                # 에러 발생 시 원본 데이터 Flatten (Fallback)
                outputs = batch_x.view(batch_x.size(0), -1)

            features_list.append(outputs.cpu().numpy())
            
    return np.concatenate(features_list, axis=0)

# -----------------------------------------------------------------------------
# 3. Anomaly Detection 실행 및 평가
# -----------------------------------------------------------------------------
def run_anomaly_detection(train_feats, test_feats, y_test, method='isolation_forest'):
    print(f"   > Running {method}...")
    
    # 모델 선언
    if method == 'isolation_forest':
        # contamination: 이상치 비율 예상값 (모르면 'auto' 혹은 작게 설정)
        clf = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    elif method == 'one_class_svm':
        clf = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
    else:
        raise ValueError("Unknown method")

    # 학습 (Train 데이터의 Feature로 학습)
    # *참고*: 비지도 학습이므로 y_train은 쓰지 않음 (혹은 Normal 데이터만 골라서 학습할 수도 있음)
    clf.fit(train_feats)

    # 예측 (1: Normal, -1: Anomaly) -> 이를 (0: Normal, 1: Anomaly)로 변환 필요
    y_pred_raw = clf.predict(test_feats)
    y_scores = clf.decision_function(test_feats) # ROC-AUC용 점수
    
    # Sklearn 출력: 1(정상), -1(이상)
    # 변환: 1 -> 0, -1 -> 1
    y_pred = np.where(y_pred_raw == 1, 0, 1)
    
    # OCSVM/IF의 decision_function은 양수일수록 정상, 음수일수록 이상인 경향이 있음
    # ROC-AUC 계산을 위해 '이상치일 확률(점수)'가 필요하므로 부호를 뒤집거나 조정
    if method == 'isolation_forest':
        # IF: 평균 경로 길이. 작을수록(음수일수록) 이상치.
        # score_samples를 쓰면 더 명확함. 여기선 decision_function 결과(-:이상, +:정상)를 반전
        score = -clf.decision_function(test_feats) 
    else:
        # OCSVM: 거리 기반. 양수(내부), 음수(외부/이상). 반전시켜서 '이상 점수'로 만듦
        score = -y_scores

    # 평가 지표 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, score)
    except:
        auc = 0.5 # 라벨이 하나뿐이거나 에러 시

    return acc, prec, rec, f1, auc

# -----------------------------------------------------------------------------
# 메인 실행 함수
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/hdd/dataset/newDataset')
    parser.add_argument('--gpu', type=int, default=0)
    # 모델 하이퍼파라미터 (저장된 모델 불러오기 위해 필요)
    parser.add_argument('--seq_len', type=int, default=96) # 데이터셋별 수정 필요
    parser.add_argument('--enc_in', type=int, default=1)   # 데이터셋별 수정 필요
    
    # 더미 인자 (PatchTST 초기화용)
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

    # 실험할 데이터셋 목록 설정
    # (DatasetName, SeqLen, EncIn)
    datasets_config = [
        ("AtrialFibrillation", 640, 2),
        ("PEMS-SF", 963, 963),
        ("StandWalkJump", 2500, 4)
    ]

    print("=== PatchTST Feature based Anomaly Detection (IF & OCSVM) ===")

    for name, seq_len, enc_in in datasets_config:
        print(f"\n[Dataset: {name}] Loading...")
        
        # 1. 데이터 로드
        X_train, y_train, X_test, y_test = load_data(name, args.root_path)
        if X_train is None: continue

        # 2. 모델 로드 (여기서는 랜덤 초기화된 모델을 사용하지만, 실제로는 학습된 체크포인트 로드 필요)
        # args 업데이트
        args.seq_len = seq_len
        args.enc_in = enc_in
        args.c_out = enc_in 
        args.num_class = 0 # Feature Extractor 모드
        
        model = PatchTST(args).float().to(device)
        
        # [중요] 학습된 가중치 불러오기 (경로가 있다면 주석 해제하여 사용)
        # checkpoint_path = f"./checkpoints/{name}_forecasting/checkpoint.pth"
        # if os.path.exists(checkpoint_path):
        #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        #     print("   > Pre-trained model loaded.")
        # else:
        #     print("   > [Warning] No checkpoint found. Using random weights.")
        
        # 3. Feature Extraction
        print("   > Extracting Features using PatchTST...")
        train_feats = extract_features(model, X_train, device)
        test_feats = extract_features(model, X_test, device)
        
        # 데이터 스케일링 (SVM 등에 중요)
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        test_feats = scaler.transform(test_feats)

        # 4. Anomaly Detection 수행 (Isolation Forest)
        acc, prec, rec, f1, auc = run_anomaly_detection(train_feats, test_feats, y_test, method='isolation_forest')
        print(f"   [Isolation Forest] Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

        # 5. Anomaly Detection 수행 (One-Class SVM)
        # PEMS-SF 처럼 데이터가 크면 OCSVM이 매우 느릴 수 있으므로 주의 (필요시 subsample)
        if name == 'PEMS-SF' and train_feats.shape[0] > 10000:
            print("   [One-Class SVM] Skipped for PEMS-SF (Too large for standard SVM)")
        else:
            acc, prec, rec, f1, auc = run_anomaly_detection(train_feats, test_feats, y_test, method='one_class_svm')
            print(f"   [One-Class SVM ] Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

    print("\n=== All Experiments Completed ===")