import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y


def load_UEA_forecast(dataset):
    """
    UEA 데이터셋을 forecasting 형식으로 로드
    
    UEA 데이터를 시계열 예측용으로 변환:
    - 모든 샘플을 연결하여 하나의 긴 시계열로 만듦
    - train/valid/test 슬라이스로 분할
    
    Returns:
        data: (1, total_timesteps, n_features) 형태의 데이터
        train_slice, valid_slice, test_slice: 데이터 분할 슬라이스
        scaler: StandardScaler 객체
        pred_lens: 예측 길이 리스트
        n_covariate_cols: covariate 컬럼 수 (0)
    """
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            res_data.append(t_data)
        return np.array(res_data).swapaxes(1, 2)  # (n_samples, n_timestamps, n_features)
    
    train_X = extract_data(train_data)
    test_X = extract_data(test_data)
    
    # 모든 샘플을 연결하여 하나의 긴 시계열로 만듦
    # (n_samples, n_timestamps, n_features) -> (total_timesteps, n_features)
    all_data = np.concatenate([train_X.reshape(-1, train_X.shape[-1]), 
                                test_X.reshape(-1, test_X.shape[-1])], axis=0)
    
    total_len = len(all_data)
    
    # 60/20/20 분할
    train_slice = slice(None, int(0.6 * total_len))
    valid_slice = slice(int(0.6 * total_len), int(0.8 * total_len))
    test_slice = slice(int(0.8 * total_len), None)
    
    # 정규화
    scaler = StandardScaler().fit(all_data[train_slice])
    all_data = scaler.transform(all_data)
    
    # (1, total_timesteps, n_features) 형태로 변환
    data = np.expand_dims(all_data, 0)
    
    # 시퀀스 길이에 따라 예측 길이 조정
    seq_len = train_X.shape[1]
    if seq_len >= 100:
        # pred_lens = [24, 48, 96]
        pred_lens = [24]
    elif seq_len >= 50:
        pred_lens = [12, 24, 48]
    else:
        pred_lens = [6, 12, 24]
    
    n_covariate_cols = 0
    
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols

    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def load_UEA_anomaly(dataset):
    """
    UEA 데이터셋을 anomaly detection 형식으로 로드
    
    UEA 데이터를 이상 탐지용으로 변환:
    - 학습 데이터는 정상 샘플(label=0)만 사용
    - 테스트 데이터에서 특정 클래스를 이상치로 간주
    - 가장 빈도가 높은 클래스를 정상, 나머지를 이상치로 처리
    - 모든 샘플을 연결하여 하나의 긴 시계열로 만듦 (기존 anomaly detection 형식과 호환)
    
    Returns:
        all_train_data: dict, 각 키는 시계열 ID, 값은 (timesteps, features) 배열
        all_train_labels: dict, 각 키는 시계열 ID, 값은 타임스텝별 레이블 (0: 정상)
        all_train_timestamps: dict, 각 키는 시계열 ID, 값은 타임스텝 인덱스
        all_test_data: dict, 위와 동일 형식
        all_test_labels: dict, 타임스텝별 레이블 (0: 정상, 1: 이상)
        all_test_timestamps: dict, 타임스텝 인덱스
        delay: int, 이상 탐지 지연 허용치 (시퀀스 길이의 10%)
    """
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    # 정규화
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    # 가장 빈도가 높은 클래스를 정상 클래스로 설정
    labels, counts = np.unique(train_y, return_counts=True)
    normal_class = labels[np.argmax(counts)]
    
    seq_len = train_X.shape[1]
    
    # 학습 데이터: 정상 클래스만 사용, 모든 샘플을 연결
    normal_mask = train_y == normal_class
    train_X_normal = train_X[normal_mask]
    
    # 모든 정상 샘플을 연결하여 하나의 긴 시계열로 만듦
    # (n_samples, timesteps, features) -> (n_samples * timesteps, features)
    train_concat = train_X_normal.reshape(-1, train_X_normal.shape[-1])
    train_labels_concat = np.zeros(len(train_concat), dtype=np.int64)
    train_timestamps_concat = np.arange(len(train_concat))
    
    # 테스트 데이터: 모든 샘플을 연결, 비정상 클래스는 이상으로 표시
    test_concat_list = []
    test_labels_list = []
    for sample, label in zip(test_X, test_y):
        test_concat_list.append(sample)
        if label == normal_class:
            test_labels_list.append(np.zeros(seq_len, dtype=np.int64))
        else:
            test_labels_list.append(np.ones(seq_len, dtype=np.int64))
    
    test_concat = np.concatenate(test_concat_list, axis=0)
    test_labels_concat = np.concatenate(test_labels_list, axis=0)
    test_timestamps_concat = np.arange(len(test_concat))
    
    # dict 형태로 변환 (load_anomaly와 동일한 형식 - 단일 키)
    key = 'series_0'
    all_train_data = {key: train_concat}
    all_train_labels = {key: train_labels_concat}
    all_train_timestamps = {key: train_timestamps_concat}
    
    all_test_data = {key: test_concat}
    all_test_labels = {key: test_labels_concat}
    all_test_timestamps = {key: test_timestamps_concat}
    
    # delay: 시퀀스 길이의 10%
    delay = max(1, seq_len // 10)
    
    return all_train_data, all_train_labels, all_train_timestamps, \
           all_test_data, all_test_labels, all_test_timestamps, \
           delay


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.stack(pretrain_data)
    # 데이터 형태 확인: 2D면 (n_samples, timesteps), 3D면 (n_samples, timesteps, features)
    if pretrain_data.ndim == 2:
        # 단변량 데이터: feature 차원 추가
        pretrain_data = np.expand_dims(pretrain_data, 2)
    # 3D인 경우 이미 (n_samples, timesteps, features) 형태
    return pretrain_data
