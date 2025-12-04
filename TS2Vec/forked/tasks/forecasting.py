import numpy as np
import time
import torch
import torch.nn as nn
import gc
from . import _eval_protocols as eval_protocols
from sklearn.metrics import r2_score

def generate_pred_samples(features, data, pred_len, drop=0):
    """
    메모리 효율적인 예측 샘플 생성
    features: (n_instances, timesteps, repr_dim)
    data: (n_instances, timesteps, n_features)
    """
    n = data.shape[1]
    features = features[:, :-pred_len]
    features = features[:, drop:]
    
    # 메모리 효율적인 방식으로 레이블 생성
    n_samples = features.shape[0]
    n_timesteps = features.shape[1]
    n_features = data.shape[2]
    
    # 각 타임스텝에서 pred_len 스텝 후의 값을 예측
    # labels shape: (n_instances, n_timesteps, pred_len, n_features)
    labels = np.zeros((n_samples, n_timesteps, pred_len, n_features), dtype=np.float32)
    for i in range(pred_len):
        start_idx = drop + 1 + i
        end_idx = start_idx + n_timesteps
        labels[:, :, i, :] = data[:, start_idx:end_idx, :]
    
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, pred_len * n_features)

def cal_metrics(pred, target):
    mse = ((pred - target) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(pred - target).mean()
    # R² 계산
    r2 = r2_score(target.flatten(), pred.flatten())
    return {
        'MSE': round(float(mse), 4),
        'RMSE': round(float(rmse), 4),
        'MAE': round(float(mae), 4),
        'R2': round(float(r2), 4)
    }


class LinearRegressor(nn.Module):
    """GPU에서 실행되는 선형 회귀 모델"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def fit_linear_gpu(train_features, train_labels, valid_features, valid_labels, device, 
                   lr=0.01, epochs=100, batch_size=1024):
    """
    GPU에서 선형 회귀 모델 학습
    """
    input_dim = train_features.shape[1]
    output_dim = train_labels.shape[1]
    
    model = LinearRegressor(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # numpy to tensor
    train_X = torch.FloatTensor(train_features).to(device)
    train_y = torch.FloatTensor(train_labels).to(device)
    valid_X = torch.FloatTensor(valid_features).to(device)
    valid_y = torch.FloatTensor(valid_labels).to(device)
    
    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(valid_X)
            val_loss = criterion(val_pred, valid_y).item()
        model.train()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, use_gpu=True):
    padding = 200
    device = model.device if hasattr(model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    t = time.time()
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    # 원본 데이터 메모리 해제
    del all_repr
    gc.collect()
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    
    for pred_len in pred_lens:
        print(f"  Processing pred_len={pred_len}...")
        
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        t = time.time()
        if use_gpu and torch.cuda.is_available():
            # GPU 기반 선형 회귀
            lr = fit_linear_gpu(train_features, train_labels, valid_features, valid_labels, device)
            lr_train_time[pred_len] = time.time() - t
            
            t = time.time()
            lr.eval()
            
            # 배치 처리로 메모리 효율성 향상
            batch_size = 1024
            test_pred_list = []
            with torch.no_grad():
                for i in range(0, len(test_features), batch_size):
                    batch = torch.FloatTensor(test_features[i:i+batch_size]).to(device)
                    pred = lr(batch).cpu().numpy()
                    test_pred_list.append(pred)
                    del batch
            test_pred = np.concatenate(test_pred_list, axis=0)
            del test_pred_list
            lr_infer_time[pred_len] = time.time() - t
            
            # GPU 메모리 정리
            del lr
            torch.cuda.empty_cache()
        else:
            # CPU 기반 Ridge 회귀 (기존 방식)
            lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
            lr_train_time[pred_len] = time.time() - t
            
            t = time.time()
            test_pred = lr.predict(test_features)
            lr_infer_time[pred_len] = time.time() - t
        
        # 학습 데이터 메모리 해제
        del train_features, train_labels, valid_features, valid_labels, test_features
        gc.collect()

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        
        # 메트릭 계산 (inverse transform 전)
        norm_metrics = cal_metrics(test_pred, test_labels)
        
        # inverse transform (메모리 효율적으로)
        if test_data.shape[0] > 1:
            test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
            test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, test_pred.shape[-1])).reshape(test_pred.shape)
            test_labels_inv = scaler.inverse_transform(test_labels.reshape(-1, test_labels.shape[-1])).reshape(test_labels.shape)
        
        raw_metrics = cal_metrics(test_pred_inv, test_labels_inv)
        
        # 결과 저장 (대용량 배열은 저장하지 않음 - 메모리 절약)
        out_log[pred_len] = {
            'norm_metrics': norm_metrics,
            'raw_metrics': raw_metrics
        }
        ours_result[pred_len] = {
            'norm': norm_metrics,
            'raw': raw_metrics
        }
        
        # 메모리 정리
        del test_pred, test_labels, test_pred_inv, test_labels_inv
        gc.collect()
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
