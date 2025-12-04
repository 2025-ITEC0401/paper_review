import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RidgeRegressor(nn.Module):
    """GPU에서 실행되는 Ridge Regression 모델"""
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def fit_ridge_gpu(repr_data, values, alpha=1.0, device='cuda', lr=0.01, epochs=100):
    """
    GPU에서 Ridge Regression 학습
    
    Args:
        repr_data: 표현 데이터 (n_samples, repr_dim)
        values: 타겟 값 (n_samples,)
        alpha: 정규화 강도
        device: GPU 디바이스
        lr: 학습률
        epochs: 에폭 수
    
    Returns:
        학습된 모델
    """
    input_dim = repr_data.shape[1]
    model = RidgeRegressor(input_dim, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
    criterion = nn.MSELoss()
    
    X = torch.FloatTensor(repr_data).to(device)
    y = torch.FloatTensor(values).unsqueeze(1).to(device)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    return model


def create_missing_mask(data, missing_ratio=0.1, missing_type='random'):
    """
    결측값 마스크 생성
    
    Args:
        data: 입력 데이터 (n_samples, n_timestamps, n_features)
        missing_ratio: 결측값 비율 (0.0 ~ 1.0)
        missing_type: 결측값 생성 방식
            - 'random': 무작위로 결측값 생성
            - 'block': 연속된 블록으로 결측값 생성
            - 'feature': 특정 feature 전체를 결측값으로
    
    Returns:
        mask: 결측값 마스크 (True: 관측값, False: 결측값)
    """
    n_samples, n_timestamps, n_features = data.shape
    mask = np.ones_like(data, dtype=bool)
    
    if missing_type == 'random':
        # 무작위로 결측값 생성
        random_mask = np.random.rand(n_samples, n_timestamps, n_features) > missing_ratio
        mask = random_mask
        
    elif missing_type == 'block':
        # 연속된 블록으로 결측값 생성
        block_length = max(1, int(n_timestamps * missing_ratio))
        for i in range(n_samples):
            for j in range(n_features):
                start_idx = np.random.randint(0, max(1, n_timestamps - block_length))
                mask[i, start_idx:start_idx + block_length, j] = False
                
    elif missing_type == 'feature':
        # 특정 시점의 모든 feature를 결측값으로
        n_missing_timestamps = max(1, int(n_timestamps * missing_ratio))
        for i in range(n_samples):
            missing_indices = np.random.choice(n_timestamps, n_missing_timestamps, replace=False)
            mask[i, missing_indices, :] = False
    
    return mask


def impute_with_representation(model, data, mask, device='cuda'):
    """
    표현 학습을 활용한 결측값 보간 (GPU 버전)
    
    Args:
        model: TS2Vec 모델
        data: 입력 데이터 (결측값은 NaN으로 표시)
        mask: 결측값 마스크 (True: 관측값, False: 결측값)
        device: GPU 디바이스
    
    Returns:
        imputed_data: 보간된 데이터
    """
    n_samples, n_timestamps, n_features = data.shape
    
    # 결측값을 NaN으로 설정
    masked_data = data.copy()
    masked_data[~mask] = np.nan
    
    # 표현 학습 (이미 GPU 사용)
    repr_data = model.encode(masked_data, encoding_window=None)  # (n_samples, n_timestamps, repr_dims)
    
    # 각 시점의 표현을 활용하여 결측값 보간
    # GPU 기반 Ridge Regression을 사용하여 관측값에서 학습 후 결측값 예측
    imputed_data = data.copy()
    
    for feature_idx in range(n_features):
        for sample_idx in range(n_samples):
            sample_mask = mask[sample_idx, :, feature_idx]
            
            if sample_mask.all():  # 결측값 없음
                continue
            if (~sample_mask).all():  # 모든 값이 결측
                continue
                
            # 관측된 시점의 표현과 값
            observed_repr = repr_data[sample_idx, sample_mask, :]
            observed_values = data[sample_idx, sample_mask, feature_idx]
            
            # 결측된 시점의 표현
            missing_repr = repr_data[sample_idx, ~sample_mask, :]
            
            if len(observed_repr) < 2:  # 학습 데이터 부족
                continue
            
            # GPU 기반 Ridge Regression으로 학습
            reg = fit_ridge_gpu(observed_repr, observed_values, alpha=1.0, device=device)
            
            # 결측값 예측 (GPU에서)
            with torch.no_grad():
                missing_repr_tensor = torch.FloatTensor(missing_repr).to(device)
                predicted_values = reg(missing_repr_tensor).squeeze().cpu().numpy()
            
            imputed_data[sample_idx, ~sample_mask, feature_idx] = predicted_values
    
    return imputed_data


def impute_with_interpolation(data, mask, method='linear'):
    """
    기본 보간법을 사용한 결측값 보간 (베이스라인)
    
    Args:
        data: 입력 데이터
        mask: 결측값 마스크
        method: 보간 방법 ('linear', 'nearest', 'mean')
    
    Returns:
        imputed_data: 보간된 데이터
    """
    from scipy import interpolate
    
    n_samples, n_timestamps, n_features = data.shape
    imputed_data = data.copy()
    
    for sample_idx in range(n_samples):
        for feature_idx in range(n_features):
            sample_mask = mask[sample_idx, :, feature_idx]
            
            if sample_mask.all():
                continue
            if (~sample_mask).all():
                imputed_data[sample_idx, :, feature_idx] = np.nanmean(data[:, :, feature_idx])
                continue
            
            observed_indices = np.where(sample_mask)[0]
            missing_indices = np.where(~sample_mask)[0]
            observed_values = data[sample_idx, sample_mask, feature_idx]
            
            if method == 'linear':
                f = interpolate.interp1d(observed_indices, observed_values, 
                                         kind='linear', fill_value='extrapolate')
                imputed_data[sample_idx, missing_indices, feature_idx] = f(missing_indices)
            elif method == 'nearest':
                f = interpolate.interp1d(observed_indices, observed_values, 
                                         kind='nearest', fill_value='extrapolate')
                imputed_data[sample_idx, missing_indices, feature_idx] = f(missing_indices)
            elif method == 'mean':
                imputed_data[sample_idx, missing_indices, feature_idx] = np.mean(observed_values)
    
    return imputed_data


def eval_imputation(model, data, missing_ratios=[0.1, 0.2, 0.3, 0.4, 0.5], 
                    missing_types=['random'], n_runs=5, device='cuda'):
    """
    Imputation 태스크 평가 (GPU 버전)
    
    평가 지표:
        - MSE (Mean Squared Error)
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - Mask Ratio (결측값 비율)
    
    Args:
        model: TS2Vec 모델
        data: 입력 데이터 (n_samples, n_timestamps, n_features)
        missing_ratios: 테스트할 결측값 비율 리스트
        missing_types: 테스트할 결측값 유형 리스트
        n_runs: 각 설정별 반복 횟수
        device: GPU 디바이스
    
    Returns:
        results: 평가 결과 딕셔너리
        summary: 요약 통계
    """
    results = []
    
    for missing_type in missing_types:
        for missing_ratio in missing_ratios:
            mse_list = []
            mae_list = []
            
            for run in range(n_runs):
                # 결측값 마스크 생성
                np.random.seed(run)
                mask = create_missing_mask(data, missing_ratio, missing_type)
                
                # 결측값이 있는 위치의 실제 값
                ground_truth = data[~mask]
                
                # TS2Vec 기반 보간 (GPU 사용)
                imputed_data = impute_with_representation(model, data, mask, device=device)
                pred = imputed_data[~mask]
                
                # 평가
                mse_list.append(mean_squared_error(ground_truth, pred))
                mae_list.append(mean_absolute_error(ground_truth, pred))
            
            # 각 mask ratio별 결과 저장
            mse = np.mean(mse_list)
            rmse = np.sqrt(mse)
            mae = np.mean(mae_list)
            
            results.append({
                'mask_ratio': missing_ratio,
                'missing_type': missing_type,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mse_std': np.std(mse_list),
                'mae_std': np.std(mae_list)
            })
    
    # 요약 통계
    summary = {
        'avg_mse': np.mean([r['mse'] for r in results]),
        'avg_rmse': np.mean([r['rmse'] for r in results]),
        'avg_mae': np.mean([r['mae'] for r in results]),
    }
    
    return results, summary


def print_imputation_results(results, summary):
    """
    Imputation 결과 출력
    """
    print("\n" + "=" * 70)
    print("Imputation Evaluation Results")
    print("=" * 70)
    print(f"{'Mask Ratio':<12} {'MSE':<20} {'RMSE':<15} {'MAE':<20}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['mask_ratio']:<12} "
              f"{r['mse']:.6f}±{r['mse_std']:.4f}   "
              f"{r['rmse']:.6f}       "
              f"{r['mae']:.6f}±{r['mae_std']:.4f}")
    
    print("-" * 70)
    print(f"{'Average':<12} {summary['avg_mse']:.6f}             "
          f"{summary['avg_rmse']:.6f}       {summary['avg_mae']:.6f}")
    print("=" * 70)
